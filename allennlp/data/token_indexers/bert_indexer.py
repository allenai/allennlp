from typing import Dict, List
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
import allennlp.data.token_indexers._bert_huggingface as bert

logger = logging.getLogger(__name__)


@TokenIndexer.register("bert")
class BertIndexer(TokenIndexer[int]):
    def __init__(self,
                 vocab_path: str = None,
                 namespace: str = 'bert',
                 unk_token: str = "[UNK]",
                 max_input_chars_per_word: int = 100) -> None:
        self._namespace = namespace
        self._added_to_vocabulary = False

        if vocab_path is None:
            logger.warning("you provided no vocabulary!")
            self.vocab: Dict[str, int] = {}
        else:
            self.vocab = bert.load_vocab(cached_path(vocab_path))

        # The BERT code itself does a two-step tokenization:
        #    sentence -> [words], and then word -> [wordpieces]
        # In AllenNLP, the first step is implemented as the ``BertSimpleWordSplitter``,
        # and this token indexer handles the second.
        self.wordpiece_tokenizer = bert.WordpieceTokenizer(self.vocab, unk_token, max_input_chars_per_word)

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If we only use pretrained models, we don't need to do anything here.
        pass

    def _add_encoding_to_vocabulary(self, vocabulary: Vocabulary) -> None:
        # pylint: disable=protected-access
        for word, idx in self.vocab.items():
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
            wordpieces = bert.convert_tokens_to_ids(self.vocab, self.wordpiece_tokenizer.tokenize(token.text))
            offset += len(wordpieces)
            offsets.append(offset)
            text_tokens.extend(wordpieces)

        return {
                index_name: text_tokens,
                f"{index_name}-offsets": offsets,
                # add mask here according to the original tokens,
                # because calling util.get_text_field_mask on the
                # "byte pair" tokens will produce the wrong shape
                "original-token-mask": [1 for _ in offsets]
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
        print(tokens, desired_num_tokens, padding_lengths)
        return {key: pad_sequence_to_length(val, desired_num_tokens[key])
                for key, val in tokens.items()}
