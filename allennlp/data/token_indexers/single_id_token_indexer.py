from typing import Dict, List, cast

from overrides import overrides

from allennlp.common.util import pad_sequence_to_length
from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers.token_indexer import TokenIndexer, TokenType


class SingleIdTokenIndexer(TokenIndexer):
    """
    This :class:`TokenIndexer` represents tokens as single integers.

    Parameters
    ----------
    namespace : ``str``, optional (default=``tokens``)
        We will use this namespace in the :class:`Vocabulary` to map strings to indices.
    lowercase_tokens : ``bool``, optional (default=``False``)
        If ``True``, we will call ``token.lower()`` before getting an index for the token from the
        vocabulary.
    """
    def __init__(self, namespace: str = 'tokens', lowercase_tokens: bool = False) -> None:
        self.namespace = namespace
        self.lowercase_tokens = lowercase_tokens

    @overrides
    def count_vocab_items(self, token: str, counter: Dict[str, Dict[str, int]]):
        if self.lowercase_tokens:
            token = token.lower()
        counter[self.namespace][token] += 1

    @overrides
    def token_to_indices(self, token: str, vocabulary: Vocabulary) -> TokenType:
        if self.lowercase_tokens:
            token = token.lower()
        return vocabulary.get_token_index(token, self.namespace)

    @overrides
    def get_input_shape(self, num_tokens: int, padding_lengths: Dict[str, int]):
        return (num_tokens,)

    @overrides
    def get_padding_token(self) -> TokenType:
        return 0

    @overrides
    def get_padding_lengths(self, token: TokenType) -> Dict[str, int]:
        return {}

    @overrides
    def pad_token_sequence(self,
                           tokens: List[TokenType],
                           desired_num_tokens: int,
                           padding_lengths: Dict[str, int]) -> List[TokenType]:
        # cast is runtime no-op that makes mypy happy
        int_tokens = cast(List[int], tokens)
        return pad_sequence_to_length(int_tokens, desired_num_tokens)

    @classmethod
    def from_params(cls, params: Params):
        """
        Parameters
        ----------
        namespace : ``str``, optional (default=``tokens``)
            We will use this namespace in the :class:`Vocabulary` to map strings to indices.
        lowercase_tokens : ``bool``, optional (default=``False``)
            If ``True``, we will call ``token.lower()`` before getting an index for the token from the
            vocabulary.
        """
        namespace = params.pop('namespace', 'tokens')
        lowercase_tokens = params.pop('lowercase_tokens', False)
        params.assert_empty(cls.__name__)
        return cls(namespace=namespace, lowercase_tokens=lowercase_tokens)
