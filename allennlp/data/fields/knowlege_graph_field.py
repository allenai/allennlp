from typing import Dict, List

from overrides import overrides

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields.field import Field, DataArray
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.knowledge_graph import KnowledgeGraph


class KnowledgeGraphField(Field[DataArray]):
    """
    ``KnowledgeGraphField`` is a ``Field`` which stores a knowledge graph representation. It indexes entities
    and collects information necessary for embedding the knowledge graph.
    """
    def __init__(self,
                 knowledge_graph: KnowledgeGraph,
                 token_indexers: Dict[str, TokenIndexer]) -> None:
        self._knowledge_graph: KnowledgeGraph = knowledge_graph
        self._token_indexers: Dict[str, TokenIndexer] = token_indexers
        # {entity: {indexer: indexed_tokens}}
        self._indexed_entities: Dict[str, Dict[str, List[int]]] = None

    @overrides
    def index(self, vocab: Vocabulary):
        entity_arrays = {}
        for entity in self._knowledge_graph.get_all_entities():
            indexed_entity = {indexer_name: indexer.token_to_indices(entity, vocab)
                              for indexer_name, indexer in self._token_indexers.items()}
            entity_arrays[entity] = indexed_entity
        self._indexed_entities = entity_arrays

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
