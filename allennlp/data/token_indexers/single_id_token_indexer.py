from typing import Dict, List

from allennlp.common.util import pad_sequence_to_length
from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers.token_indexer import TokenIndexer


class SingleIdTokenIndexer(TokenIndexer[int]):
    """
    This :class:`TokenIndexer` represents tokens as single integers.

    Parameters
    ----------
    token_namespace : ``str``, optional (default=``tokens``)
        We will use this namespace in the :class:`Vocabulary` to map strings to indices.
    lowercase_tokens : ``bool``, optional (default=``False``)
        If ``True``, we will call ``token.lower()`` before getting an index for the token from the
        vocabulary.
    """
    def __init__(self, token_namespace: str = 'tokens', lowercase_tokens: bool = False) -> None:
        self.token_namespace = token_namespace
        self.lowercase_tokens = lowercase_tokens

    # TODO(joelgrus) uncomment these once fix is merged to overrides library
    # @overrides
    def count_vocab_items(self, token: str, counter: Dict[str, Dict[str, int]]):
        if self.lowercase_tokens:
            token = token.lower()
        counter[self.token_namespace][token] += 1

    # @overrides
    def token_to_indices(self, token: str, vocabulary: Vocabulary) -> int:
        if self.lowercase_tokens:
            token = token.lower()
        return vocabulary.get_token_index(token, self.token_namespace)

    # @overrides
    def get_input_shape(self, num_tokens: int, padding_lengths: Dict[str, int]):  # pylint: disable=unused-argument,no-self-use
        return (num_tokens,)

    # @overrides
    def get_padding_token(self) -> int:  # pylint: disable=no-self-use
        return 0

    # @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:  # pylint: disable=unused-argument,no-self-use
        return {}

    # @overrides
    def pad_token_sequence(self,                                           # pylint: disable=no-self-use
                           tokens: List[int],
                           desired_num_tokens: int,
                           padding_lengths: Dict[str, int]) -> List[int]:  # pylint: disable=unused-argument
        return pad_sequence_to_length(tokens, desired_num_tokens)

    @classmethod
    def from_params(cls, params: Params):
        """
        Parameters
        ----------
        token_namespace : ``str``, optional (default=``tokens``)
            We will use this namespace in the :class:`Vocabulary` to map strings to indices.
        lowercase_tokens : ``bool``, optional (default=``False``)
            If ``True``, we will call ``token.lower()`` before getting an index for the token from the
            vocabulary.
        """
        token_namespace = params.pop('token_namespace', 'tokens')
        lowercase_tokens = params.pop('lowercase_tokens', False)
        params.assert_empty(cls.__name__)
        return cls(token_namespace=token_namespace, lowercase_tokens=lowercase_tokens)
