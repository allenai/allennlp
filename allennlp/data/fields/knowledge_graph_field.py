"""
``KnowledgeGraphField`` is a ``Field`` which stores a knowledge graph representation.
"""
from typing import Callable, Dict, List, Set

import editdistance
from overrides import overrides
import torch
from torch.autograd import Variable

from allennlp.common import util
from allennlp.common.checks import ConfigurationError
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
          "linking": torch.Tensor  # shape (batch_size, num_entities, num_utterance_tokens, num_features)
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
    utterance_tokens : ``List[Token]``
        The tokens in some utterance that is paired with the ``KnowledgeGraph``.  We compute a set
        of features for linking tokens in the utterance to entities in the graph.
    tokenizer : ``Tokenizer``
        We'll use this ``Tokenizer`` to tokenize the text representation of each entity.
    token_indexers : ``Dict[str, TokenIndexer]``
        Token indexers that convert entities into arrays, similar to how text tokens are treated in
        a ``TextField``.  These might operate on the name of the entity itself, its type, its
        neighbors in the graph, etc.
    feature_extractors : ``List[str]``, optional
        Names of feature extractors to use for computing linking features.  These must be
        attributes of this object, without the first underscore.  The feature extraction functions
        are listed as the last methods in this class.  For example, to use
        :func:`_exact_token_match`, you would pass the string ``exact_token_match``.  We will add
        an underscore and look for a function matching that name.  If this list is omitted, we will
        use all available feature functions.
    """
    def __init__(self,
                 knowledge_graph: KnowledgeGraph,
                 utterance_tokens: List[Token],
                 token_indexers: Dict[str, TokenIndexer],
                 tokenizer: Tokenizer = None,
                 feature_extractors: List[str] = None,
                 entity_tokens: List[List[Token]] = None,
                 linking_features: List[List[List[float]]] = None) -> None:
        self.knowledge_graph = knowledge_graph
        if not entity_tokens:
            entity_texts = [knowledge_graph.entity_text[entity] for entity in knowledge_graph.entities]
            # TODO(mattg): Because we do tagging on each of these entities in addition to just
            # tokenizations, this is quite slow, and about half of our data processing time just
            # goes to this (~15 minutes when there are 7k instances).  The reason we do tagging is
            # so that we can add lemma features.  If we can remove the need for lemma / other
            # hand-written features, like with a CNN, we can cut down our data processing time by a
            # factor of 2.
            self.entity_texts = tokenizer.batch_tokenize(entity_texts)
        else:
            self.entity_texts = entity_tokens
        self.utterance_tokens = utterance_tokens
        self._token_indexers: Dict[str, TokenIndexer] = token_indexers
        self._indexed_entity_texts: Dict[str, TokenList] = None

        feature_extractors = feature_extractors or [
                'exact_token_match',
                'contains_exact_token_match',
                'lemma_match',
                'contains_lemma_match',
                'edit_distance',
                'related_column',
                'related_column_lemma',
                ]
        self._feature_extractors: List[Callable[[str, List[Token], Token], float]] = []
        for feature_extractor_name in feature_extractors:
            extractor = getattr(self, '_' + feature_extractor_name, None)
            if not extractor:
                raise ConfigurationError(f"Invalid feature extractor name: {feature_extractor_name}")
            self._feature_extractors.append(extractor)

        if not linking_features:
            # For quicker lookups in our feature functions, we'll additionally store some
            # dictionaries that map entity strings to useful information about the entity.
            self._entity_text_map: Dict[str, List[Token]] = {}
            for entity, entity_text in zip(knowledge_graph.entities, self.entity_texts):
                self._entity_text_map[entity] = entity_text

            self._entity_text_exact_text: Dict[str, Set[str]] = {}
            for entity, entity_text in zip(knowledge_graph.entities, self.entity_texts):
                self._entity_text_exact_text[entity] = set(e.text for e in entity_text)

            self._entity_text_lemmas: Dict[str, Set[str]] = {}
            for entity, entity_text in zip(knowledge_graph.entities, self.entity_texts):
                self._entity_text_lemmas[entity] = set(e.lemma_ for e in entity_text)
            self.linking_features = self._compute_linking_features()
        else:
            self.linking_features = linking_features

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
        padding_lengths = {'num_entities': len(self.entity_texts),
                           'num_utterance_tokens': len(self.utterance_tokens)}
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
        desired_num_utterance_tokens = padding_lengths['num_utterance_tokens']
        for indexer_name, indexer in self._token_indexers.items():
            padded_entities = util.pad_sequence_to_length(self._indexed_entity_texts[indexer_name],
                                                          desired_num_entities,
                                                          default_value=lambda: [])
            padded_arrays = []
            for padded_entity in padded_entities:
                padded_array = indexer.pad_token_sequence(padded_entity,
                                                          desired_num_entity_tokens,
                                                          padding_lengths)
                padded_arrays.append(padded_array)
            tensor = Variable(torch.LongTensor(padded_arrays), volatile=not for_training)
            tensors[indexer_name] = tensor if cuda_device == -1 else tensor.cuda(cuda_device)
        padded_linking_features = util.pad_sequence_to_length(self.linking_features,
                                                              desired_num_entities,
                                                              default_value=lambda: [])
        padded_linking_arrays = []
        default_feature_value = lambda: [0.0] * len(self._feature_extractors)
        for linking_features in padded_linking_features:
            padded_features = util.pad_sequence_to_length(linking_features,
                                                          desired_num_utterance_tokens,
                                                          default_value=default_feature_value)
            padded_linking_arrays.append(padded_features)
        linking_features_tensor = Variable(torch.FloatTensor(padded_linking_arrays),
                                           volatile=not for_training)
        if cuda_device != -1:
            linking_features_tensor = linking_features_tensor.cuda(cuda_device)
        return {'text': tensors, 'linking': linking_features_tensor}

    def _compute_linking_features(self) -> List[List[List[float]]]:
        linking_features = []
        for entity, entity_text in zip(self.knowledge_graph.entities, self.entity_texts):
            entity_features = []
            for token in self.utterance_tokens:
                token_features = []
                for feature_extractor in self._feature_extractors:
                    token_features.append(feature_extractor(entity, entity_text, token))
                entity_features.append(token_features)
            linking_features.append(entity_features)
        return linking_features

    @overrides
    def empty_field(self) -> 'KnowledgeGraphField':
        return KnowledgeGraphField(KnowledgeGraph(set(), {}), [], self._token_indexers)

    @overrides
    def batch_tensors(self, tensor_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # pylint: disable=no-self-use
        batched_text = nn_util.batch_tensor_dicts(tensor['text'] for tensor in tensor_list)  # type: ignore
        batched_linking = torch.stack([tensor['linking'] for tensor in tensor_list])
        return {'text': batched_text, 'linking': batched_linking}

    # Below here we have feature extractor functions.  To keep a consistent API for easy logic
    # above, some of these functions have unused arguments.
    # For the feature functions used in the original parser written in PNP, see here:
    # https://github.com/allenai/pnp/blob/wikitables2/src/main/scala/org/allenai/wikitables/SemanticParserFeatureGenerator.scala
    # pylint: disable=unused-argument,no-self-use

    def _exact_token_match(self, entity: str, entity_text: List[Token], token: Token) -> float:
        if len(entity_text) != 1:
            return 0.0
        return self._contains_exact_token_match(entity, entity_text, token)

    def _contains_exact_token_match(self, entity: str, entity_text: List[Token], token: Token) -> float:
        if token.text in self._entity_text_exact_text[entity]:
            return 1.0
        return 0.0

    def _lemma_match(self, entity: str, entity_text: List[Token], token: Token) -> float:
        if len(entity_text) != 1:
            return 0.0
        return self._contains_lemma_match(entity, entity_text, token)

    def _contains_lemma_match(self, entity: str, entity_text: List[Token], token: Token) -> float:
        if token.text in self._entity_text_exact_text[entity]:
            return 1.0
        if token.lemma_ in self._entity_text_lemmas[entity]:
            return 1.0
        return 0.0

    def _edit_distance(self, entity: str, entity_text: List[Token], token: Token) -> float:
        edit_distance = float(editdistance.eval(' '.join(e.text for e in entity_text), token.text))
        return 1.0 - edit_distance / len(token.text)

    def _related_column(self, entity: str, entity_text: List[Token], token: Token) -> float:
        if not entity.startswith('fb:row.row'):
            return 0.0
        for neighbor in self.knowledge_graph.neighbors[entity]:
            if token.text in self._entity_text_exact_text[neighbor]:
                return 1.0
        return 0.0

    def _related_column_lemma(self, entity: str, entity_text: List[Token], token: Token) -> float:
        if not entity.startswith('fb:row.row'):
            return 0.0
        for neighbor in self.knowledge_graph.neighbors[entity]:
            if token.text in self._entity_text_exact_text[neighbor]:
                return 1.0
            if token.lemma_ in self._entity_text_lemmas[neighbor]:
                return 1.0
        return 0.0

    # pylint: enable=unused-argument,no-self-use
