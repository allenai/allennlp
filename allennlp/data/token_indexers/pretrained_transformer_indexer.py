from typing import Dict, List
import itertools

from overrides import overrides
import pytorch_transformers
from pytorch_transformers.tokenization_utils import PreTrainedTokenizer
import torch

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer


@TokenIndexer.register("pretrained_transformer")
class PretrainedTransformerIndexer(TokenIndexer[int]):
    """
    This :class:`TokenIndexer` uses a tokenizer from the ``pytorch_transformers`` repository to
    index tokens.

    Parameters
    ----------
    model_name : ``str``
        The name of the ``pytorch_transformers`` model to use.
    namespace : ``str``, optional (default=``tokens``)
        We will add the tokens in the pytorch_transformer vocabulary to this vocabulary namespace.
    """
    # pylint: disable=no-self-use
    def __init__(self,
                 model_name: str,
                 namespace: str = "tokens",
                 token_min_padding_length: int = 0) -> None:
        super().__init__(token_min_padding_length)
        self.tokenizer = PreTrainedTokenizer.from_pretrained(model_name)
        self._namespace = namespace
        self._added_to_vocabulary = False

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If we only use pretrained models, we don't need to do anything here.
        pass

    def _add_encoding_to_vocabulary(self, vocabulary: Vocabulary) -> None:
        # pylint: disable=protected-access
        for word, idx in self.tokenizer.vocab.items():
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
        token_text = [token.text for token in tokens]
        indices = self.tokenizer.convert_tokens_to_ids(token_text)

        return {index_name: indices}

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
