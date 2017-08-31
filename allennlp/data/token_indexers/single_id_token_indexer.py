from typing import Dict, List

from overrides import overrides

from allennlp.common.util import pad_sequence_to_length
from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers.token_indexer import TokenIndexer



@TokenIndexer.register("single_id")
class SingleIdTokenIndexer(TokenIndexer[int]):
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
    # pylint: disable=no-self-use
    def __init__(self, namespace: str = 'tokens', lowercase_tokens: bool = False) -> None:
        self.namespace = namespace
        self.lowercase_tokens = lowercase_tokens

    @overrides
    def count_vocab_items(self, token: str, counter: Dict[str, Dict[str, int]]):
        # If this is used with a CharacterTokenizer that's doing byte encoding, the token might
        # already be an int.  In that case, we'll just bypass the vocabulary entirely.
        if not isinstance(token, int):
            if self.lowercase_tokens:
                token = token.lower()
            counter[self.namespace][token] += 1

    @overrides
    def token_to_indices(self, token: str, vocabulary: Vocabulary) -> int:
        # If this is used with a CharacterTokenizer that's doing byte encoding, the token might
        # already be an int.  In that case, we'll just bypass the vocabulary entirely.
        if isinstance(token, int):
            index = token
        else:
            if self.lowercase_tokens:
                token = token.lower()
            index = vocabulary.get_token_index(token, self.namespace)
        return index

    @overrides
    def get_input_shape(self, num_tokens: int, padding_lengths: Dict[str, int]):  # pylint: disable=unused-argument
        return (num_tokens,)

    @overrides
    def get_padding_token(self) -> int:
        return 0

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    @overrides
    def pad_token_sequence(self,
                           tokens: List[int],
                           desired_num_tokens: int,
                           padding_lengths: Dict[str, int]) -> List[int]:  # pylint: disable=unused-argument
        return pad_sequence_to_length(tokens, desired_num_tokens)

    @classmethod
    def from_params(cls, params: Params) -> 'SingleIdTokenIndexer':
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
