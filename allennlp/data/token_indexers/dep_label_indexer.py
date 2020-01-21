import logging
from typing import Dict, List, Set

from overrides import overrides

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer, IndexedTokenList

logger = logging.getLogger(__name__)


@TokenIndexer.register("dependency_label")
class DepLabelIndexer(TokenIndexer):
    """
    This :class:`TokenIndexer` represents tokens by their syntactic dependency label, as determined
    by the `dep_` field on `Token`.

    # Parameters

    namespace : `str`, optional (default=`dep_labels`)
        We will use this namespace in the :class:`Vocabulary` to map strings to indices.
    token_min_padding_length : `int`, optional (default=`0`)
        See :class:`TokenIndexer`.
    """

    def __init__(self, namespace: str = "dep_labels", token_min_padding_length: int = 0) -> None:
        super().__init__(token_min_padding_length)
        self.namespace = namespace
        self._logged_errors: Set[str] = set()

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        dep_label = token.dep_
        if not dep_label:
            if token.text not in self._logged_errors:
                logger.warning("Token had no dependency label: %s", token.text)
                self._logged_errors.add(token.text)
            dep_label = "NONE"
        counter[self.namespace][dep_label] += 1

    @overrides
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary
    ) -> Dict[str, List[int]]:
        dep_labels = [token.dep_ or "NONE" for token in tokens]

        return {
            "tokens": [
                vocabulary.get_token_index(dep_label, self.namespace) for dep_label in dep_labels
            ]
        }

    @overrides
    def get_empty_token_list(self) -> IndexedTokenList:
        return {"tokens": []}
