import logging
from typing import Dict, List, Set

from overrides import overrides

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer, IndexedTokenList

logger = logging.getLogger(__name__)


@TokenIndexer.register("pos_tag")
class PosTagIndexer(TokenIndexer):
    """
    This :class:`TokenIndexer` represents tokens by their part of speech tag, as determined by
    the `pos_` or `tag_` fields on `Token` (corresponding to spacy's coarse-grained and
    fine-grained POS tags, respectively).

    # Parameters

    namespace : `str`, optional (default=`pos_tokens`)
        We will use this namespace in the :class:`Vocabulary` to map strings to indices.
    coarse_tags : `bool`, optional (default=`False`)
        If `True`, we will use coarse POS tags instead of the default fine-grained POS tags.
    token_min_padding_length : `int`, optional (default=`0`)
        See :class:`TokenIndexer`.
    """

    def __init__(
        self,
        namespace: str = "pos_tokens",
        coarse_tags: bool = False,
        token_min_padding_length: int = 0,
    ) -> None:
        super().__init__(token_min_padding_length)
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
            tag = "NONE"
        counter[self._namespace][tag] += 1

    @overrides
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary
    ) -> Dict[str, List[int]]:
        tags: List[str] = []

        for token in tokens:
            if self._coarse_tags:
                tag = token.pos_
            else:
                tag = token.tag_
            if not tag:
                tag = "NONE"

            tags.append(tag)

        return {"tokens": [vocabulary.get_token_index(tag, self._namespace) for tag in tags]}

    @overrides
    def get_empty_token_list(self) -> IndexedTokenList:
        return {"tokens": []}
