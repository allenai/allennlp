from typing import Dict, List
import json
import tarfile

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer


@TokenIndexer.register("openai_transformer_byte_pair")
class OpenaiTransformerBytePairIndexer(TokenIndexer[List[int]]):
    """
    This is unlike most of our TokenIndexers in that its
    indexing is not based on a `Vocabulary` but on a fixed
    set of mappings that are loaded by the constructor.
    """
    # pylint: disable=no-self-use
    def __init__(self,
                 transformer_model_path: str,
                 n_ctx: int = 512) -> None:

        # if `file_path` is a URL, redirect to the cache
        transformer_model_path = cached_path(transformer_model_path)

        with tarfile.open(transformer_model_path) as tmp:
            encoder_info = tmp.extractfile('model/encoder_bpe_40000.json')
            self.encoder = json.load(encoder_info)

            bpe_info = tmp.extractfile('model/vocab_40000.bpe')
            # First line is "version", last line is blank
            lines = bpe_info.read().decode('utf-8').split('\n')[1:-1]

        self.decoder = {word_id: word for word, word_id in self.encoder.items()}

        # Convert "b1 b2" -> (b1, b2)
        pairs = [tuple(line.split()) for line in lines]
        # Compute ranks
        self.bpe_ranks = {pair: idx for idx, pair in enumerate(pairs)}

        self.cache: Dict[str, List[int]] = {}
        self.n_ctx = n_ctx

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If we only use pretrained models, I don't know that we need to do anything here.
        pass

    def byte_pair_encode(self, token: Token, lowercase: bool = True) -> List[str]:
        if lowercase:
            text = token.text.lower()
        else:
            text = token.text

        if text in self.cache:
            return self.cache[text]

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


    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          _vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        text_tokens = []
        offsets = []
        offset = -1

        for token in tokens:
            bpe_tokens = [self.encoder.get(t, 0) for t in self.byte_pair_encode(token)]
            offset += len(bpe_tokens)
            offsets.append(offset)
            text_tokens.extend(bpe_tokens)

        num_tokens = len(text_tokens)
        if num_tokens < self.n_ctx:
            text_tokens.extend(0 for _ in range(self.n_ctx - num_tokens))
        else:
            text_tokens = text_tokens[:self.n_ctx]

        return {
                index_name: text_tokens,
                f"{index_name}-offsets": offsets,
                # add mask here according to the original tokens,
                # because calling util.get_text_field_mask on the
                # "byte pair" tokens will produce the wrong shape
                "mask": [1 for _ in tokens]
        }

    @overrides
    def get_padding_token(self) -> int:
        return 0

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    @overrides
    def pad_token_sequence(self,
                           tokens: Dict[str, List[int]],
                           desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, List[int]]:  # pylint: disable=unused-argument
        return {key: pad_sequence_to_length(val, desired_num_tokens[key])
                for key, val in tokens.items()}
