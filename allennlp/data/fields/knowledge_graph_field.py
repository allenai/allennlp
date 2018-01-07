"""
``KnowledgeGraphField`` is a ``Field`` which stores a knowledge graph representation.
"""
from typing import Dict, List

import torch
from torch.autograd import Variable
from overrides import overrides

from allennlp.data.fields.field import Field
from allennlp.data.semparse import KnowledgeGraph
from allennlp.data.token_indexers.token_indexer import TokenIndexer, TokenType
from allennlp.data.tokenizers.token import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn import util

TokenList = List[TokenType]  # pylint: disable=invalid-name


class KnowledgeGraphField(Field[Dict[str, torch.Tensor]]):
    """
    A ``KnowledgeGraphField`` represents a ``KnowledgeGraph`` as a ``Field`` that can be used in a
    ``Model``.  We take the (sorted) list of entities in the graph and output them as arrays using
    ``TokenIndexers``, similar to how text tokens are treated by a ``TextField``.  We have
    knowledge-graph-specific ``TokenIndexers``, however, that allow for more versatile treatment of
    the knowledge graph entities than just treating them as text tokens.

    Parameters
    ----------
    knowledge_graph : ``KnowledgeGraph``
        The knowledge graph that this field stores.
    token_indexers : ``Dict[str, TokenIndexer]``
        Token indexers that convert entities into arrays, similar to how text tokens are treated in
        a ``TextField``.  These might operate on the name of the entity itself, its type, its
        neighbors in the graph, etc.
    """
    def __init__(self,
                 knowledge_graph: KnowledgeGraph,
                 token_indexers: Dict[str, TokenIndexer]) -> None:
        self.knowledge_graph: KnowledgeGraph = knowledge_graph
        self.entities = [Token(entity) for entity in sorted(self.knowledge_graph.get_all_entities())]
        self._token_indexers: Dict[str, TokenIndexer] = token_indexers
        self._indexed_entities: Dict[str, TokenList] = None

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for indexer in self._token_indexers.values():
            for entity in self.entities:
                indexer.count_vocab_items(entity, counter)

    @overrides
    def index(self, vocab: Vocabulary):
        entity_arrays = {}
        for indexer_name, indexer in self._token_indexers.items():
            arrays = [indexer.token_to_indices(entity, vocab) for entity in self.entities]
            entity_arrays[indexer_name] = arrays
        self._indexed_entities = entity_arrays

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        lengths = []
        assert self._indexed_entities is not None, ("This field is not indexed yet. Call "
                                                    ".index(vocabulary) before determining padding "
                                                    "lengths.")
        for indexer_name, indexer in self._token_indexers.items():
            indexer_lengths = {}

            # This is a list of dicts, one for each token in the field.
            entity_lengths = [indexer.get_padding_lengths(entity)
                              for entity in self._indexed_entities[indexer_name]]
            # Iterate over the keys in the first element of the list.  This is fine as for a given
            # indexer, all entities will return the same keys, so we can just use the first one.
            for key in entity_lengths[0].keys():
                indexer_lengths[key] = max(x[key] if key in x else 0 for x in entity_lengths)
            lengths.append(indexer_lengths)

        any_indexed_entity_key = list(self._indexed_entities.keys())[0]
        padding_lengths = {'num_entities': len(self._indexed_entities[any_indexed_entity_key])}

        # Get all the keys which have been used for padding.
        padding_keys = {key for d in lengths for key in d.keys()}
        for padding_key in padding_keys:
            padding_lengths[padding_key] = max(x[padding_key] if padding_key in x else 0 for x in lengths)
        return padding_lengths

    @overrides
    def as_tensor(self,
                  padding_lengths: Dict[str, int],
                  cuda_device: int = -1,
                  for_training: bool = True) -> Dict[str, torch.Tensor]:
        tensors = {}
        desired_num_entities = padding_lengths['num_entities']
        for indexer_name, indexer in self._token_indexers.items():
            padded_array = indexer.pad_token_sequence(self._indexed_entities[indexer_name],
                                                      desired_num_entities, padding_lengths)
            # Use the key of the indexer to recognise what the array corresponds to within the field
            # (i.e. the result of word indexing, or the result of character indexing, for example).
            tensor = Variable(torch.LongTensor(padded_array), volatile=not for_training)
            tensors[indexer_name] = tensor if cuda_device == -1 else tensor.cuda(cuda_device)
        return tensors

    @overrides
    def empty_field(self) -> 'KnowledgeGraphField':
        return KnowledgeGraphField(KnowledgeGraph({}), self._token_indexers)

    @overrides
    def batch_tensors(self, tensor_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # pylint: disable=no-self-use
        return util.batch_tensor_dicts(tensor_list)
