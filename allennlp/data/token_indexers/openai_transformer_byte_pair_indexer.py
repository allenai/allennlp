from typing import Dict, List, Tuple
import json
import tarfile
import re

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer


def text_standardize(text):
    """
    Apply text standardization following original implementation.
    """
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('…', '...')
    text = text.replace('´', "'")
    text = re.sub(r'''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
    text = re.sub(r'\s*\n\s*', ' \n ', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text.strip()


@TokenIndexer.register("openai_transformer_byte_pair")
class OpenaiTransformerBytePairIndexer(TokenIndexer[int]):
    """
    Generates the indices for the byte-pair encoding used by
    the OpenAI transformer language model: https://blog.openai.com/language-unsupervised/

    This is unlike most of our TokenIndexers in that its
    indexing is not based on a `Vocabulary` but on a fixed
    set of mappings that are loaded by the constructor.

    Note: recommend using ``OpenAISplitter`` tokenizer with this indexer,
    as it applies the same text normalization as the original implementation.

    Note 2: when ``tokens_to_add`` is not None, be sure to set
    ``n_special=len(tokens_to_add)`` in ``OpenaiTransformer``, otherwise
    behavior is undefined.
    """
    # pylint: disable=no-self-use
    def __init__(self,
                 encoder: Dict[str, int] = None,
                 byte_pairs: List[Tuple[str, str]] = None,
                 n_ctx: int = 512,
                 model_path: str = None,
                 namespace: str = 'openai_transformer',
                 tokens_to_add: List[str] = None,
                 token_min_padding_length: int = 0) -> None:
        super().__init__(token_min_padding_length)
        self._namespace = namespace
        self._added_to_vocabulary = False

        too_much_information = model_path and (encoder or byte_pairs)
        too_little_information = not model_path and not (encoder and byte_pairs)

        if too_much_information or too_little_information:
            raise ConfigurationError("must specify either model path or (encoder + byte_pairs) but not both")

        if model_path:
            model_path = cached_path(model_path)

            # Load encoder and byte_pairs from tar.gz
            with tarfile.open(model_path) as tmp:
                encoder_name = next(m.name for m in tmp.getmembers() if 'encoder_bpe' in m.name)
                encoder_info = tmp.extractfile(encoder_name)

                if encoder_info:
                    encoder = json.loads(encoder_info.read())
                else:
                    raise ConfigurationError(f"expected encoder_bpe file in archive {model_path}")

                bpe_name = next(m.name for m in tmp.getmembers() if m.name.endswith('.bpe'))
                bpe_info = tmp.extractfile(bpe_name)

                if bpe_info:
                    # First line is "version", last line is blank
                    lines = bpe_info.read().decode('utf-8').split('\n')[1:-1]
                    # Convert "b1 b2" -> (b1, b2)
                    byte_pairs = [tuple(line.split()) for line in lines]  # type: ignore
                else:
                    raise ConfigurationError(f"expected .bpe file in archive {model_path}")

        if tokens_to_add is not None:
            for token in tokens_to_add:
                encoder[token + '</w>'] = len(encoder)
            self.tokens_to_add = set(tokens_to_add)
        else:
            self.tokens_to_add = None

        self.encoder = encoder
        self.decoder = {word_id: word for word, word_id in self.encoder.items()}

        # Compute ranks
        self.bpe_ranks = {pair: idx for idx, pair in enumerate(byte_pairs)}

        self.cache: Dict[str, List[str]] = {}
        self.n_ctx = n_ctx

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If we only use pretrained models, we don't need to do anything here.
        pass

    def byte_pair_encode(self, token: Token, lowercase: bool = True) -> List[str]:
        if lowercase:
            text = token.text.lower()
        else:
            text = token.text

        if text in self.cache:
            return self.cache[text]

        if self.tokens_to_add and text in self.tokens_to_add:
            # this is a special token, and it's guaranteed to be a word
            word = [text + '</w>']
            self.cache[text] = word
            return word

        # Split into letters, but add a `</w>` to the last
        word = [c for c in text[:-1]]
        word.append(text[-1] + '</w>')

        # Get unique pairs (prev_symbol, next_symbol)
        pairs = {(prev_symbol, next_symbol)
                 for prev_symbol, next_symbol in zip(word, word[1:])}

        if not pairs:
            return [text + '</w>']

        while True:
            # Find the highest ranked pair
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))

            # If that pair is not actually ranked, stop.
            if bigram not in self.bpe_ranks:
                break

            # Split up the pair
            first, second = bigram

            # and make a helper for a new word
            new_word = []
            i = 0

            # Iterate over the letters of the word
            while i < len(word):
                try:
                    # Find first occurrence of `first` after i,
                    j = word.index(first, i)
                    # add all the characters preceding it,
                    new_word.extend(word[i:j])
                    # and update i to j
                    i = j
                except ValueError:
                    # `first` didn't occur, so just add the rest
                    new_word.extend(word[i:])
                    break  # out of while i < len(word)

                # At this point we know word[i] == first
                if i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = new_word
            if len(word) == 1:
                break  # out of while True
            else:
                pairs = {(prev_symbol, next_symbol)
                         for prev_symbol, next_symbol in zip(word, word[1:])}

        if ' '.join(word) == '\n  </w>':
            word = ['\n</w>']

        self.cache[text] = word
        return word

    def _add_encoding_to_vocabulary(self, vocabulary: Vocabulary) -> None:
        # pylint: disable=protected-access
        for word, idx in self.encoder.items():
            vocabulary._token_to_index[self._namespace][word] = idx
            vocabulary._index_to_token[self._namespace][idx] = word

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        if not self._added_to_vocabulary:
            self._add_encoding_to_vocabulary(vocabulary)
            self._added_to_vocabulary = True

        text_tokens = []
        offsets = []
        offset = -1

        for token in tokens:
            bpe_tokens = [self.encoder.get(t, 0) for t in self.byte_pair_encode(token)]
            offset += len(bpe_tokens)
            offsets.append(offset)
            text_tokens.extend(bpe_tokens)

        num_tokens = len(text_tokens)

        # If there's too many tokens, that's going to cause problems.
        if num_tokens > self.n_ctx:
            raise RuntimeError(f"The transformer model has a maximum sequence length of {self.n_ctx} "
                               f"but your byte pair encoded sequence has length {num_tokens}. "
                               f"The offending text input is {tokens}.")

        # If there's too few tokens, just pad with zeros.
        text_tokens.extend(0 for _ in range(self.n_ctx - num_tokens))

        return {
                index_name: text_tokens,
                f"{index_name}-offsets": offsets,
                # add mask here according to the original tokens,
                # because calling util.get_text_field_mask on the
                # "byte pair" tokens will produce the wrong shape
                "mask": [1 for _ in offsets]
        }

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    @overrides
    def as_padded_tensor(self,
                         tokens: Dict[str, List[int]],
                         desired_num_tokens: Dict[str, int],
                         padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:  # pylint: disable=unused-argument
        return {key: torch.LongTensor(pad_sequence_to_length(val, desired_num_tokens[key]))
                for key, val in tokens.items()}
