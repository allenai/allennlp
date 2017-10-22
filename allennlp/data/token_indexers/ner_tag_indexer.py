import logging
from typing import Dict, List

from overrides import overrides

from allennlp.common.util import pad_sequence_to_length
from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TokenIndexer.register("ner_tag")
class NerTagIndexer(TokenIndexer[int]):
    """
    This :class:`TokenIndexer` represents tokens by their entity type (i.e., their NER tag), as
    determined by the ``ent_type_`` field on ``Token``.

    Parameters
    ----------
    namespace : ``str``, optional (default=``ner_tags``)
        We will use this namespace in the :class:`Vocabulary` to map strings to indices.
    """
    # pylint: disable=no-self-use
    def __init__(self, namespace: str = 'ner_tags') -> None:
        self._namespace = namespace

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        tag = token.ent_type_
        if not tag:
            tag = 'NONE'
        counter[self._namespace][tag] += 1

    @overrides
    def token_to_indices(self, token: Token, vocabulary: Vocabulary) -> int:
        tag = token.ent_type_
        if tag is None:
            tag = 'NONE'
        return vocabulary.get_token_index(tag, self._namespace)

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
    def from_params(cls, params: Params) -> 'NerTagIndexer':
        namespace = params.pop('namespace', 'ner_tags')
        params.assert_empty(cls.__name__)
        return cls(namespace=namespace)
