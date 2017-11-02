from typing import Dict, List, Optional

from overrides import overrides

from allennlp.data.fields.field import Field, DataArray
from allennlp.data.token_indexers.token_indexer import TokenIndexer, TokenType
from allennlp.data.knowledge_graph import KnowledgeGraph


class KnowledgeGraphField(Field[DataArray]):
    """
    ``KnowledgeGraphField`` is a ``Field`` which stores a knowledge graph representation. It indexes entities
    and collects information necessary for embedding the knowledge graph.
    """
    def __init__(self,
                 knowledge_graph: KnowledgeGraph,
                 token_indexers: Dict[str, TokenIndexer]) -> None:
        self._knowledge_graph = knowledge_graph
        self._token_indexers = token_indexers
        self._indexed_entities: Optional[Dict[str, List[TokenType]]] = None

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        # pylint: disable=no-self-use
        return {}

    @overrides
    def as_array(self, padding_lengths: Dict[str, int]) -> DataArray:
        # pylint: disable=unused-argument
        return self._indexed_entities

    @overrides
    def empty_field(self) -> 'KnowledgeGraphField':
        return KnowledgeGraphField(KnowledgeGraph({}), self._token_indexers)
