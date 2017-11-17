"""
``KnowledgeGraphField`` is a ``Field`` which stores a knowledge graph representation.
"""

from typing import Dict, List

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields.field import Field, DataArray
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.semparse.knowledge_graphs import KnowledgeGraph


class KnowledgeGraphField(Field[DataArray]):
    """
    ``KnowledgeGraphField`` indexes entities and collects information necessary for embedding the knowledge
    graph. Here we assume the token indexers will index all the information needed to embed entities
    (say names and types).

    Parameters
    ----------
    knowledge_graph : ``KnowledgeGraph``
        The knowledge graph that this field stores.
    token_indexers : ``Dict[str, TokenIndexer]``
        Token indexers for indexing various aspects of entities (say names and type of entities) for embedding
        them.
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
        # TODO (pradeep): This may change after the actual entity token indexers are implemented.
        if self._indexed_entities is None:
            raise ConfigurationError("This field is not indexed yet. Call .index(vocabulary) before determining "
                                     "padding lengths.")
        padding_lengths = {"num_tokens": len(self._indexed_entities)}
        for indexer_name in self._token_indexers:
            padding_lengths[indexer_name] = max([len(index[indexer_name]) if indexer_name in index else 0
                                                 for index in self._indexed_entities.values()])
        return padding_lengths

    @overrides
    def as_array(self, padding_lengths: Dict[str, int]) -> DataArray:
        # pylint: disable=unused-argument
        return self._indexed_entities

    @overrides
    def empty_field(self) -> 'KnowledgeGraphField':
        return KnowledgeGraphField(KnowledgeGraph({}), self._token_indexers)
