import logging
from typing import Dict, List, Set

from overrides import overrides

from allennlp.common.util import pad_sequence_to_length
from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TokenIndexer.register("pos_tag")
class PosTagIndexer(TokenIndexer[int]):
    """
    This :class:`TokenIndexer` represents tokens by their part of speech tag, as determined by
    the ``pos_`` or ``tag_`` fields on ``Token`` (corresponding to spacy's coarse-grained and
    fine-grained POS tags, respectively).

    Parameters
    ----------
    namespace : ``str``, optional (default=``pos_tags``)
        We will use this namespace in the :class:`Vocabulary` to map strings to indices.
    coarse_tags : ``bool``, optional (default=``False``)
        If ``True``, we will use coarse POS tags instead of the default fine-grained POS tags.
    """
    # pylint: disable=no-self-use
    def __init__(self, namespace: str = 'pos_tags', coarse_tags: bool = False) -> None:
        self._namespace = namespace
        self._coarse_tags = coarse_tags
        self._logged_errors: Set[str] = set()

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        if self._coarse_tags:
            tag = token.pos_
        else:
            tag = token.tag_
        if not tag:
            if token.text not in self._logged_errors:
                logger.warning("Token had no POS tag: %s", token.text)
                self._logged_errors.add(token.text)
            tag = 'NONE'
        counter[self._namespace][tag] += 1

    @overrides
    def token_to_indices(self, token: Token, vocabulary: Vocabulary) -> int:
        if self._coarse_tags:
            tag = token.pos_
        else:
            tag = token.tag_
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
    def from_params(cls, params: Params) -> 'PosTagIndexer':
        namespace = params.pop('namespace', 'pos_tags')
        coarse_tags = params.pop_bool('coarse_tags', False)
        params.assert_empty(cls.__name__)
        return cls(namespace=namespace, coarse_tags=coarse_tags)
