"""
``KnowledgeGraphField`` is a ``Field`` which stores a knowledge graph representation.
"""
from typing import Dict, List

import torch
from torch.autograd import Variable
from overrides import overrides

from allennlp.common import util
from allennlp.data.fields.field import Field
from allennlp.data.semparse import KnowledgeGraph
from allennlp.data.token_indexers.token_indexer import TokenIndexer, TokenType
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn import util as nn_util

TokenList = List[TokenType]  # pylint: disable=invalid-name


class KnowledgeGraphField(Field[Dict[str, torch.Tensor]]):
    """
    A ``KnowledgeGraphField`` represents a ``KnowledgeGraph`` as a ``Field`` that can be used in a
    ``Model``.  For each entity in the graph, we output two things: a text representation of the
    entity, handled identically to a ``TextField``, and a list of linking features for each token
    in some input utterance.

    The output of this field is a dictionary::

        {
          "text": Dict[str, torch.Tensor],  # each tensor has shape (batch_size, num_entities, num_entity_tokens)
          "linking": torch.Tensor  # shape (batch_size, num_entities, num_utterance_tokens)
        }

    The ``text`` component of this dictionary is suitable to be passed into a
    ``TextFieldEmbedder`` (which handles the additional ``num_entities`` dimension without any
    issues).  The ``linking`` component of the dictionary can be used however you want to decide
    which tokens in the utterance correspond to which entities in the knowledge graph.

    In order to create the ``text`` component, we use the same dictionary of ``TokenIndexers``
    that's used in a ``TextField`` (as we're just representing the text corresponding to each
    entity).  For the ``linking`` component, we use a set of hard-coded feature extractors that
    operate between the text corresponding to each entity and each token in the utterance.

    Parameters
    ----------
    knowledge_graph : ``KnowledgeGraph``
        The knowledge graph that this field stores.
    tokenizer : ``Tokenizer``
        We'll use this ``Tokenizer`` to tokenize the text representation of each entity.
    token_indexers : ``Dict[str, TokenIndexer]``
        Token indexers that convert entities into arrays, similar to how text tokens are treated in
        a ``TextField``.  These might operate on the name of the entity itself, its type, its
        neighbors in the graph, etc.
    """
    def __init__(self,
                 knowledge_graph: KnowledgeGraph,
                 tokenizer: Tokenizer,
                 token_indexers: Dict[str, TokenIndexer]) -> None:
        self.knowledge_graph = knowledge_graph
        self.entity_texts = [tokenizer.tokenize(knowledge_graph.entity_text[entity])
                            for entity in knowledge_graph.entities]
        self._tokenizer = tokenizer
        self._token_indexers: Dict[str, TokenIndexer] = token_indexers
        self._indexed_entity_texts: Dict[str, TokenList] = None

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for indexer in self._token_indexers.values():
            for entity_text in self.entity_texts:
                for token in entity_text:
                    indexer.count_vocab_items(token, counter)

    @overrides
    def index(self, vocab: Vocabulary):
        self._indexed_entity_texts = {}
        for indexer_name, indexer in self._token_indexers.items():
            indexer_arrays = []
            for entity_text in self.entity_texts:
                indexer_arrays.append([indexer.token_to_indices(token, vocab) for token in entity_text])
            self._indexed_entity_texts[indexer_name] = indexer_arrays

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        padding_lengths = {'num_entities': len(self.entity_texts)}
        padding_lengths['num_entity_tokens'] = max(len(entity_text) for entity_text in self.entity_texts)
        lengths = []
        assert self._indexed_entity_texts is not None, ("This field is not indexed yet. Call "
                                                       ".index(vocab) before determining padding "
                                                       "lengths.")
        for indexer_name, indexer in self._token_indexers.items():
            indexer_lengths = {}

            # This is a list of dicts, one for each token in the field.
            entity_lengths = [indexer.get_padding_lengths(token)
                              for entity_text in self._indexed_entity_texts[indexer_name]
                              for token in entity_text]
            # Iterate over the keys in the first element of the list.  This is fine as for a given
            # indexer, all entities will return the same keys, so we can just use the first one.
            for key in entity_lengths[0].keys():
                indexer_lengths[key] = max(x[key] if key in x else 0 for x in entity_lengths)
            lengths.append(indexer_lengths)

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
        desired_num_entity_tokens = padding_lengths['num_entity_tokens']
        for indexer_name, indexer in self._token_indexers.items():
            padded_entities = util.pad_sequence_to_length(self._indexed_entity_texts[indexer_name],
                                                          desired_num_entities,
                                                          default_value=lambda x: [])
            padded_arrays = []
            for padded_entity in padded_entities:
                padded_array = indexer.pad_token_sequence(padded_entity,
                                                          desired_num_entity_tokens,
                                                          padding_lengths)
                print(padded_array)
                padded_arrays.append(padded_array)
            print(padded_arrays)
            tensor = Variable(torch.LongTensor(padded_arrays), volatile=not for_training)
            print(tensor)
            tensors[indexer_name] = tensor if cuda_device == -1 else tensor.cuda(cuda_device)
        return {'text': tensors}

    @overrides
    def empty_field(self) -> 'KnowledgeGraphField':
        return KnowledgeGraphField(KnowledgeGraph(set(), {}), self._tokenizer, self._token_indexers)

    @overrides
    def batch_tensors(self, tensor_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # pylint: disable=no-self-use
        batched_text = nn_util.batch_tensor_dicts(tensor['text'] for tensor in tensor_list)
        return {'text': batched_text}
