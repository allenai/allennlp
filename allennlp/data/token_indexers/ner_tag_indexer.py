import logging
from typing import Dict, List

from overrides import overrides

from allennlp.common.util import pad_sequence_to_length
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
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        tags = ['NONE' if token.ent_type_ is None else token.ent_type_ for token in tokens]

        return {index_name: [vocabulary.get_token_index(tag, self._namespace) for tag in tags]}

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
