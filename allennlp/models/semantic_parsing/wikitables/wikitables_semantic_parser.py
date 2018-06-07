from typing import Dict, List, Tuple

from overrides import overrides
import torch

from allennlp.common.checks import check_dimensions_match
from allennlp.common.util import pad_sequence_to_length
from allennlp.data import Vocabulary
from allennlp.data.fields.production_rule_field import ProductionRuleArray
from allennlp.models.model import Model
from allennlp.models.semantic_parsing.wikitables.wikitables_decoder_state import WikiTablesDecoderState
from allennlp.modules import Embedding, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder, TimeDistributed
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn import util
from allennlp.nn.decoding import GrammarState, RnnState, ChecklistState
from allennlp.semparse.type_declarations import type_declaration
from allennlp.semparse.type_declarations.type_declaration import START_SYMBOL
from allennlp.semparse.worlds import WikiTablesWorld
from allennlp.training.metrics import Average, WikiTablesAccuracy


class WikiTablesSemanticParser(Model):
    """
    A ``WikiTablesSemanticParser`` is a :class:`Model` which takes as input a table and a question,
    and produces a logical form that answers the question when executed over the table.  The
    logical form is generated by a `type-constrained`, `transition-based` parser. This is an
    abstract class that defines most of the functionality related to the transition-based parser. It
    does not contain the implementation for actually training the parser. You may want to train it
    using a learning-to-search algorithm, in which case you will want to use
    ``WikiTablesErmSemanticParser``, or if you have a set of approximate logical forms that give the
    correct denotation, you will want to use ``WikiTablesMmlSemanticParser``.

    Parameters
    ----------
    vocab : ``Vocabulary``
    question_embedder : ``TextFieldEmbedder``
        Embedder for questions.
    action_embedding_dim : ``int``
        Dimension to use for action embeddings.
    encoder : ``Seq2SeqEncoder``
        The encoder to use for the input question.
    entity_encoder : ``Seq2VecEncoder``
        The encoder to used for averaging the words of an entity.
    max_decoding_steps : ``int``
        When we're decoding with a beam search, what's the maximum number of steps we should take?
        This only applies at evaluation time, not during training.
    use_neighbor_similarity_for_linking : ``bool``, optional (default=False)
        If ``True``, we will compute a max similarity between a question token and the `neighbors`
        of an entity as a component of the linking scores.  This is meant to capture the same kind
        of information as the ``related_column`` feature.
    dropout : ``float``, optional (default=0)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer).
    num_linking_features : ``int``, optional (default=10)
        We need to construct a parameter vector for the linking features, so we need to know how
        many there are.  The default of 8 here matches the default in the ``KnowledgeGraphField``,
        which is to use all eight defined features. If this is 0, another term will be added to the
        linking score. This term contains the maximum similarity value from the entity's neighbors
        and the question.
    rule_namespace : ``str``, optional (default=rule_labels)
        The vocabulary namespace to use for production rules.  The default corresponds to the
        default used in the dataset reader, so you likely don't need to modify this.
    tables_directory : ``str``, optional (default=/wikitables/)
        The directory to find tables when evaluating logical forms.  We rely on a call to SEMPRE to
        evaluate logical forms, and SEMPRE needs to read the table from disk itself.  This tells
        SEMPRE where to find the tables.
    """
    # pylint: disable=abstract-method
    def __init__(self,
                 vocab: Vocabulary,
                 question_embedder: TextFieldEmbedder,
                 action_embedding_dim: int,
                 encoder: Seq2SeqEncoder,
                 entity_encoder: Seq2VecEncoder,
                 max_decoding_steps: int,
                 use_neighbor_similarity_for_linking: bool = False,
                 dropout: float = 0.0,
                 num_linking_features: int = 10,
                 rule_namespace: str = 'rule_labels',
                 tables_directory: str = '/wikitables/') -> None:
        super(WikiTablesSemanticParser, self).__init__(vocab)
        self._question_embedder = question_embedder
        self._encoder = encoder
        self._entity_encoder = TimeDistributed(entity_encoder)
        self._max_decoding_steps = max_decoding_steps
        self._use_neighbor_similarity_for_linking = use_neighbor_similarity_for_linking
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._rule_namespace = rule_namespace
        self._denotation_accuracy = WikiTablesAccuracy(tables_directory)
        self._action_sequence_accuracy = Average()
        self._has_logical_form = Average()

        self._action_padding_index = -1  # the padding value used by IndexField
        num_actions = vocab.get_vocab_size(self._rule_namespace)
        self._action_embedder = Embedding(num_embeddings=num_actions, embedding_dim=action_embedding_dim)
        self._output_action_embedder = Embedding(num_embeddings=num_actions, embedding_dim=action_embedding_dim)
        self._action_biases = Embedding(num_embeddings=num_actions, embedding_dim=1)

        # This is what we pass as input in the first step of decoding, when we don't have a
        # previous action, or a previous question attention.
        self._first_action_embedding = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))
        self._first_attended_question = torch.nn.Parameter(torch.FloatTensor(encoder.get_output_dim()))
        torch.nn.init.normal_(self._first_action_embedding)
        torch.nn.init.normal_(self._first_attended_question)

        check_dimensions_match(entity_encoder.get_output_dim(), question_embedder.get_output_dim(),
                               "entity word average embedding dim", "question embedding dim")

        self._num_entity_types = 4  # TODO(mattg): get this in a more principled way somehow?
        self._num_start_types = 5  # TODO(mattg): get this in a more principled way somehow?
        self._embedding_dim = question_embedder.get_output_dim()
        self._type_params = torch.nn.Linear(self._num_entity_types, self._embedding_dim)
        self._neighbor_params = torch.nn.Linear(self._embedding_dim, self._embedding_dim)

        if num_linking_features > 0:
            self._linking_params = torch.nn.Linear(num_linking_features, 1)
        else:
            self._linking_params = None

        if self._use_neighbor_similarity_for_linking:
            self._question_entity_params = torch.nn.Linear(1, 1)
            self._question_neighbor_params = torch.nn.Linear(1, 1)
        else:
            self._question_entity_params = None
            self._question_neighbor_params = None

    def _get_initial_state_and_scores(self,
                                      question: Dict[str, torch.LongTensor],
                                      table: Dict[str, torch.LongTensor],
                                      world: List[WikiTablesWorld],
                                      actions: List[List[ProductionRuleArray]],
                                      example_lisp_string: List[str] = None,
                                      add_world_to_initial_state: bool = False,
                                      checklist_states: List[ChecklistState] = None) -> Dict:
        """
        Does initial preparation and creates an intiial state for both the semantic parsers. Note
        that the checklist state is optional, and the ``WikiTablesMmlParser`` is not expected to
        pass it.
        """
        table_text = table['text']
        # (batch_size, question_length, embedding_dim)
        embedded_question = self._question_embedder(question)
        question_mask = util.get_text_field_mask(question).float()
        # (batch_size, num_entities, num_entity_tokens, embedding_dim)
        embedded_table = self._question_embedder(table_text, num_wrapping_dims=1)
        table_mask = util.get_text_field_mask(table_text, num_wrapping_dims=1).float()

        batch_size, num_entities, num_entity_tokens, _ = embedded_table.size()
        num_question_tokens = embedded_question.size(1)

        # (batch_size, num_entities, embedding_dim)
        encoded_table = self._entity_encoder(embedded_table, table_mask)
        # (batch_size, num_entities, num_neighbors)
        neighbor_indices = self._get_neighbor_indices(world, num_entities, encoded_table)

        # Neighbor_indices is padded with -1 since 0 is a potential neighbor index.
        # Thus, the absolute value needs to be taken in the index_select, and 1 needs to
        # be added for the mask since that method expects 0 for padding.
        # (batch_size, num_entities, num_neighbors, embedding_dim)
        embedded_neighbors = util.batched_index_select(encoded_table, torch.abs(neighbor_indices))

        neighbor_mask = util.get_text_field_mask({'ignored': neighbor_indices + 1},
                                                 num_wrapping_dims=1).float()

        # Encoder initialized to easily obtain a masked average.
        neighbor_encoder = TimeDistributed(BagOfEmbeddingsEncoder(self._embedding_dim, averaged=True))
        # (batch_size, num_entities, embedding_dim)
        embedded_neighbors = neighbor_encoder(embedded_neighbors, neighbor_mask)

        # entity_types: one-hot tensor with shape (batch_size, num_entities, num_types)
        # entity_type_dict: Dict[int, int], mapping flattened_entity_index -> type_index
        # These encode the same information, but for efficiency reasons later it's nice
        # to have one version as a tensor and one that's accessible on the cpu.
        entity_types, entity_type_dict = self._get_type_vector(world, num_entities, encoded_table)

        entity_type_embeddings = self._type_params(entity_types.float())
        projected_neighbor_embeddings = self._neighbor_params(embedded_neighbors.float())
        # (batch_size, num_entities, embedding_dim)
        entity_embeddings = torch.nn.functional.tanh(entity_type_embeddings + projected_neighbor_embeddings)


        # Compute entity and question word cosine similarity. Need to add a small value to
        # to the table norm since there are padding values which cause a divide by 0.
        embedded_table = embedded_table / (embedded_table.norm(dim=-1, keepdim=True) + 1e-13)
        embedded_question = embedded_question / (embedded_question.norm(dim=-1, keepdim=True) + 1e-13)
        question_entity_similarity = torch.bmm(embedded_table.view(batch_size,
                                                                   num_entities * num_entity_tokens,
                                                                   self._embedding_dim),
                                               torch.transpose(embedded_question, 1, 2))

        question_entity_similarity = question_entity_similarity.view(batch_size,
                                                                     num_entities,
                                                                     num_entity_tokens,
                                                                     num_question_tokens)

        # (batch_size, num_entities, num_question_tokens)
        question_entity_similarity_max_score, _ = torch.max(question_entity_similarity, 2)

        # (batch_size, num_entities, num_question_tokens, num_features)
        linking_features = table['linking']

        linking_scores = question_entity_similarity_max_score

        if self._use_neighbor_similarity_for_linking:
            # The linking score is computed as a linear projection of two terms. The first is the
            # maximum similarity score over the entity's words and the question token. The second
            # is the maximum similarity over the words in the entity's neighbors and the question
            # token.
            #
            # The second term, projected_question_neighbor_similarity, is useful when a column
            # needs to be selected. For example, the question token might have no similarity with
            # the column name, but is similar with the cells in the column.
            #
            # Note that projected_question_neighbor_similarity is intended to capture the same
            # information as the related_column feature.
            #
            # Also note that this block needs to be _before_ the `linking_params` block, because
            # we're overwriting `linking_scores`, not adding to it.

            # (batch_size, num_entities, num_neighbors, num_question_tokens)
            question_neighbor_similarity = util.batched_index_select(question_entity_similarity_max_score,
                                                                     torch.abs(neighbor_indices))
            # (batch_size, num_entities, num_question_tokens)
            question_neighbor_similarity_max_score, _ = torch.max(question_neighbor_similarity, 2)
            projected_question_entity_similarity = self._question_entity_params(
                    question_entity_similarity_max_score.unsqueeze(-1)).squeeze(-1)
            projected_question_neighbor_similarity = self._question_neighbor_params(
                    question_neighbor_similarity_max_score.unsqueeze(-1)).squeeze(-1)
            linking_scores = projected_question_entity_similarity + projected_question_neighbor_similarity

        feature_scores = None
        if self._linking_params is not None:
            feature_scores = self._linking_params(linking_features).squeeze(3)
            linking_scores = linking_scores + feature_scores

        # (batch_size, num_question_tokens, num_entities)
        linking_probabilities = self._get_linking_probabilities(world, linking_scores.transpose(1, 2),
                                                                question_mask, entity_type_dict)

        # (batch_size, num_question_tokens, embedding_dim)
        link_embedding = util.weighted_sum(entity_embeddings, linking_probabilities)
        encoder_input = torch.cat([link_embedding, embedded_question], 2)

        # (batch_size, question_length, encoder_output_dim)
        encoder_outputs = self._dropout(self._encoder(encoder_input, question_mask))

        # This will be our initial hidden state and memory cell for the decoder LSTM.
        final_encoder_output = util.get_final_encoder_states(encoder_outputs,
                                                             question_mask,
                                                             self._encoder.is_bidirectional())
        memory_cell = encoder_outputs.new_zeros(batch_size, self._encoder.get_output_dim())

        initial_score = embedded_question.data.new_zeros(batch_size)

        action_embeddings, output_action_embeddings, action_biases, action_indices = self._embed_actions(actions)

        _, num_entities, num_question_tokens = linking_scores.size()
        flattened_linking_scores, actions_to_entities = self._map_entity_productions(linking_scores,
                                                                                     world,
                                                                                     actions)
        # To make grouping states together in the decoder easier, we convert the batch dimension in
        # all of our tensors into an outer list.  For instance, the encoder outputs have shape
        # `(batch_size, question_length, encoder_output_dim)`.  We need to convert this into a list
        # of `batch_size` tensors, each of shape `(question_length, encoder_output_dim)`.  Then we
        # won't have to do any index selects, or anything, we'll just do some `torch.cat()`s.
        initial_score_list = [initial_score[i] for i in range(batch_size)]
        encoder_output_list = [encoder_outputs[i] for i in range(batch_size)]
        question_mask_list = [question_mask[i] for i in range(batch_size)]
        initial_rnn_state = []
        for i in range(batch_size):
            initial_rnn_state.append(RnnState(final_encoder_output[i],
                                              memory_cell[i],
                                              self._first_action_embedding,
                                              self._first_attended_question,
                                              encoder_output_list,
                                              question_mask_list))
        initial_grammar_state = [self._create_grammar_state(world[i], actions[i])
                                 for i in range(batch_size)]
        initial_state_world = world if add_world_to_initial_state else None
        initial_state = WikiTablesDecoderState(batch_indices=list(range(batch_size)),
                                               action_history=[[] for _ in range(batch_size)],
                                               score=initial_score_list,
                                               rnn_state=initial_rnn_state,
                                               grammar_state=initial_grammar_state,
                                               action_embeddings=action_embeddings,
                                               output_action_embeddings=output_action_embeddings,
                                               action_biases=action_biases,
                                               action_indices=action_indices,
                                               possible_actions=actions,
                                               flattened_linking_scores=flattened_linking_scores,
                                               actions_to_entities=actions_to_entities,
                                               entity_types=entity_type_dict,
                                               world=initial_state_world,
                                               example_lisp_string=example_lisp_string,
                                               checklist_state=checklist_states,
                                               debug_info=None)
        return {"initial_state": initial_state,
                "linking_scores": linking_scores,
                "feature_scores": feature_scores,
                "similarity_scores": question_entity_similarity_max_score}

    @staticmethod
    def _get_neighbor_indices(worlds: List[WikiTablesWorld],
                              num_entities: int,
                              tensor: torch.Tensor) -> torch.LongTensor:
        """
        This method returns the indices of each entity's neighbors. A tensor
        is accepted as a parameter for copying purposes.

        Parameters
        ----------
        worlds : ``List[WikiTablesWorld]``
        num_entities : ``int``
        tensor : ``torch.Tensor``
            Used for copying the constructed list onto the right device.

        Returns
        -------
        A ``torch.LongTensor`` with shape ``(batch_size, num_entities, num_neighbors)``. It is padded
        with -1 instead of 0, since 0 is a valid neighbor index.
        """

        num_neighbors = 0
        for world in worlds:
            for entity in world.table_graph.entities:
                if len(world.table_graph.neighbors[entity]) > num_neighbors:
                    num_neighbors = len(world.table_graph.neighbors[entity])

        batch_neighbors = []
        for world in worlds:
            # Each batch instance has its own world, which has a corresponding table.
            entities = world.table_graph.entities
            entity2index = {entity: i for i, entity in enumerate(entities)}
            entity2neighbors = world.table_graph.neighbors
            neighbor_indexes = []
            for entity in entities:
                entity_neighbors = [entity2index[n] for n in entity2neighbors[entity]]
                # Pad with -1 instead of 0, since 0 represents a neighbor index.
                padded = pad_sequence_to_length(entity_neighbors, num_neighbors, lambda: -1)
                neighbor_indexes.append(padded)
            neighbor_indexes = pad_sequence_to_length(neighbor_indexes,
                                                      num_entities,
                                                      lambda: [-1] * num_neighbors)
            batch_neighbors.append(neighbor_indexes)
        return tensor.new_tensor(batch_neighbors, dtype=torch.long)

    @staticmethod
    def _get_type_vector(worlds: List[WikiTablesWorld],
                         num_entities: int,
                         tensor: torch.Tensor) -> Tuple[torch.LongTensor, Dict[int, int]]:
        """
        Produces the one hot encoding for each entity's type. In addition,
        a map from a flattened entity index to type is returned to combine
        entity type operations into one method.

        Parameters
        ----------
        worlds : ``List[WikiTablesWorld]``
        num_entities : ``int``
        tensor : ``torch.Tensor``
            Used for copying the constructed list onto the right device.

        Returns
        -------
        A ``torch.LongTensor`` with shape ``(batch_size, num_entities, num_types)``.
        entity_types : ``Dict[int, int]``
            This is a mapping from ((batch_index * num_entities) + entity_index) to entity type id.
        """
        entity_types = {}
        batch_types = []
        for batch_index, world in enumerate(worlds):
            types = []
            for entity_index, entity in enumerate(world.table_graph.entities):
                one_hot_vectors = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                # We need numbers to be first, then cells, then parts, then row, because our
                # entities are going to be sorted.  We do a split by type and then a merge later,
                # and it relies on this sorting.
                if entity.startswith('fb:cell'):
                    entity_type = 1
                elif entity.startswith('fb:part'):
                    entity_type = 2
                elif entity.startswith('fb:row'):
                    entity_type = 3
                else:
                    entity_type = 0
                types.append(one_hot_vectors[entity_type])

                # For easier lookups later, we're actually using a _flattened_ version
                # of (batch_index, entity_index) for the key, because this is how the
                # linking scores are stored.
                flattened_entity_index = batch_index * num_entities + entity_index
                entity_types[flattened_entity_index] = entity_type
            padded = pad_sequence_to_length(types, num_entities, lambda: [0, 0, 0, 0])
            batch_types.append(padded)
        return tensor.new_tensor(batch_types), entity_types

    def _get_linking_probabilities(self,
                                   worlds: List[WikiTablesWorld],
                                   linking_scores: torch.FloatTensor,
                                   question_mask: torch.LongTensor,
                                   entity_type_dict: Dict[int, int]) -> torch.FloatTensor:
        """
        Produces the probability of an entity given a question word and type. The logic below
        separates the entities by type since the softmax normalization term sums over entities
        of a single type.

        Parameters
        ----------
        worlds : ``List[WikiTablesWorld]``
        linking_scores : ``torch.FloatTensor``
            Has shape (batch_size, num_question_tokens, num_entities).
        question_mask: ``torch.LongTensor``
            Has shape (batch_size, num_question_tokens).
        entity_type_dict : ``Dict[int, int]``
            This is a mapping from ((batch_index * num_entities) + entity_index) to entity type id.

        Returns
        -------
        batch_probabilities : ``torch.FloatTensor``
            Has shape ``(batch_size, num_question_tokens, num_entities)``.
            Contains all the probabilities for an entity given a question word.
        """
        _, num_question_tokens, num_entities = linking_scores.size()
        batch_probabilities = []

        for batch_index, world in enumerate(worlds):
            all_probabilities = []
            num_entities_in_instance = 0

            # NOTE: The way that we're doing this here relies on the fact that entities are
            # implicitly sorted by their types when we sort them by name, and that numbers come
            # before "fb:cell", and "fb:cell" comes before "fb:row".  This is not a great
            # assumption, and could easily break later, but it should work for now.
            for type_index in range(self._num_entity_types):
                # This index of 0 is for the null entity for each type, representing the case where a
                # word doesn't link to any entity.
                entity_indices = [0]
                entities = world.table_graph.entities
                for entity_index, _ in enumerate(entities):
                    if entity_type_dict[batch_index * num_entities + entity_index] == type_index:
                        entity_indices.append(entity_index)

                if len(entity_indices) == 1:
                    # No entities of this type; move along...
                    continue

                # We're subtracting one here because of the null entity we added above.
                num_entities_in_instance += len(entity_indices) - 1

                # We separate the scores by type, since normalization is done per type.  There's an
                # extra "null" entity per type, also, so we have `num_entities_per_type + 1`.  We're
                # selecting from a (num_question_tokens, num_entities) linking tensor on _dimension 1_,
                # so we get back something of shape (num_question_tokens,) for each index we're
                # selecting.  All of the selected indices together then make a tensor of shape
                # (num_question_tokens, num_entities_per_type + 1).
                indices = linking_scores.new_tensor(entity_indices, dtype=torch.long)
                entity_scores = linking_scores[batch_index].index_select(1, indices)

                # We used index 0 for the null entity, so this will actually have some values in it.
                # But we want the null entity's score to be 0, so we set that here.
                entity_scores[:, 0] = 0

                # No need for a mask here, as this is done per batch instance, with no padding.
                type_probabilities = torch.nn.functional.softmax(entity_scores, dim=1)
                all_probabilities.append(type_probabilities[:, 1:])

            # We need to add padding here if we don't have the right number of entities.
            if num_entities_in_instance != num_entities:
                zeros = linking_scores.new_zeros(num_question_tokens,
                                                 num_entities - num_entities_in_instance)
                all_probabilities.append(zeros)

            # (num_question_tokens, num_entities)
            probabilities = torch.cat(all_probabilities, dim=1)
            batch_probabilities.append(probabilities)
        batch_probabilities = torch.stack(batch_probabilities, dim=0)
        return batch_probabilities * question_mask.unsqueeze(-1).float()

    @staticmethod
    def _action_history_match(predicted: List[int], targets: torch.LongTensor) -> int:
        # TODO(mattg): this could probably be moved into a FullSequenceMatch metric, or something.
        # Check if target is big enough to cover prediction (including start/end symbols)
        if len(predicted) > targets.size(1):
            return 0
        predicted_tensor = targets.new_tensor(predicted)
        targets_trimmed = targets[:, :len(predicted)]
        # Return 1 if the predicted sequence is anywhere in the list of targets.
        return torch.max(torch.min(targets_trimmed.eq(predicted_tensor), dim=1)[0]).item()

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        We track three metrics here:

            1. dpd_acc, which is the percentage of the time that our best output action sequence is
            in the set of action sequences provided by DPD.  This is an easy-to-compute lower bound
            on denotation accuracy for the set of examples where we actually have DPD output.  We
            only score dpd_acc on that subset.

            2. denotation_acc, which is the percentage of examples where we get the correct
            denotation.  This is the typical "accuracy" metric, and it is what you should usually
            report in an experimental result.  You need to be careful, though, that you're
            computing this on the full data, and not just the subset that has DPD output (make sure
            you pass "keep_if_no_dpd=True" to the dataset reader, which we do for validation data,
            but not training data).

            3. lf_percent, which is the percentage of time that decoding actually produces a
            finished logical form.  We might not produce a valid logical form if the decoder gets
            into a repetitive loop, or we're trying to produce a super long logical form and run
            out of time steps, or something.
        """
        return {
                'dpd_acc': self._action_sequence_accuracy.get_metric(reset),
                'denotation_acc': self._denotation_accuracy.get_metric(reset),
                'lf_percent': self._has_logical_form.get_metric(reset),
                }

    @staticmethod
    def _create_grammar_state(world: WikiTablesWorld,
                              possible_actions: List[ProductionRuleArray]) -> GrammarState:
        valid_actions = world.get_valid_actions()
        action_mapping = {}
        for i, action in enumerate(possible_actions):
            action_string = action[0]
            action_mapping[action_string] = i
        translated_valid_actions = {}
        for key, action_strings in valid_actions.items():
            translated_valid_actions[key] = [action_mapping[action_string]
                                             for action_string in action_strings]
        return GrammarState([START_SYMBOL],
                            {},
                            translated_valid_actions,
                            action_mapping,
                            type_declaration.is_nonterminal)

    def _embed_actions(self, actions: List[List[ProductionRuleArray]]) -> Tuple[torch.Tensor,
                                                                                torch.Tensor,
                                                                                torch.Tensor,
                                                                                Dict[Tuple[int, int], int]]:
        """
        Given all of the possible actions for all batch instances, produce an embedding for them.
        There will be significant overlap in this list, as the production rules from the grammar
        are shared across all batch instances.  Our returned tensor has an embedding for each
        `unique` action, so we also need to return a mapping from the original ``(batch_index,
        action_index)`` to our new ``global_action_index``, so that we can get the right action
        embedding during decoding.

        Returns
        -------
        action_embeddings : ``torch.Tensor``
            Has shape ``(num_unique_actions, action_embedding_dim)``.
        output_action_embeddings : ``torch.Tensor``
            Has shape ``(num_unique_actions, action_embedding_dim)``.
        action_biases : ``torch.Tensor``
            Has shape ``(num_unique_actions, 1)``.
        action_map : ``Dict[Tuple[int, int], int]``
            Maps ``(batch_index, action_index)`` in the input action list to ``action_index`` in
            the ``action_embeddings`` tensor.  All non-embeddable actions get mapped to `-1` here.
        """
        # TODO(mattg): This whole action pipeline might be a whole lot more complicated than it
        # needs to be.  We used to embed actions differently (using some crazy ideas about
        # embedding the LHS and RHS separately); we could probably get away with simplifying things
        # further now that we're just doing a simple embedding for global actions.  But I'm leaving
        # it like this for now to have a minimal change to go from the LHS/RHS embedding to a
        # single action embedding.
        embedded_actions = self._action_embedder.weight
        output_embedded_actions = self._output_action_embedder.weight
        action_biases = self._action_biases.weight

        # Now we just need to make a map from `(batch_index, action_index)` to
        # `global_action_index`.  global_action_ids has the list of all unique actions; here we're
        # going over all of the actions for each batch instance so we can map them to the global
        # action ids.
        action_vocab = self.vocab.get_token_to_index_vocabulary(self._rule_namespace)
        action_map: Dict[Tuple[int, int], int] = {}
        for batch_index, instance_actions in enumerate(actions):
            for action_index, action in enumerate(instance_actions):
                if not action[0]:
                    # This rule is padding.
                    continue
                global_action_id = action_vocab.get(action[0], -1)
                action_map[(batch_index, action_index)] = global_action_id
        return embedded_actions, output_embedded_actions, action_biases, action_map

    @staticmethod
    def _map_entity_productions(linking_scores: torch.FloatTensor,
                                worlds: List[WikiTablesWorld],
                                actions: List[List[ProductionRuleArray]]) -> Tuple[torch.Tensor,
                                                                                   Dict[Tuple[int, int], int]]:
        """
        Constructs a map from ``(batch_index, action_index)`` to ``(batch_index * entity_index)``.
        That is, some actions correspond to terminal productions of entities from our table.  We
        need to find those actions and map them to their corresponding entity indices, where the
        entity index is its position in the list of entities returned by the ``world``.  This list
        is what defines the second dimension of the ``linking_scores`` tensor, so we can use this
        index to look up linking scores for each action in that tensor.

        For easier processing later, the mapping that we return is `flattened` - we really want to
        map ``(batch_index, action_index)`` to ``(batch_index, entity_index)``, but we are going to
        have to use the result of this mapping to do ``index_selects`` on the ``linking_scores``
        tensor.  You can't do ``index_select`` with tuples, so we flatten ``linking_scores`` to
        have shape ``(batch_size * num_entities, num_question_tokens)``, and return shifted indices
        into this flattened tensor.

        Parameters
        ----------
        linking_scores : ``torch.Tensor``
            A tensor representing linking scores between each table entity and each question token.
            Has shape ``(batch_size, num_entities, num_question_tokens)``.
        worlds : ``List[WikiTablesWorld]``
            The ``World`` for each batch instance.  The ``World`` contains a reference to the
            ``TableKnowledgeGraph`` that defines the set of entities in the linking.
        actions : ``List[List[ProductionRuleArray]]``
            The list of possible actions for each batch instance.  Our action indices are defined
            in terms of this list, so we'll find entity productions in this list and map them to
            entity indices from the entity list we get from the ``World``.

        Returns
        -------
        flattened_linking_scores : ``torch.Tensor``
            A flattened version of ``linking_scores``, with shape ``(batch_size * num_entities,
            num_question_tokens)``.
        actions_to_entities : ``Dict[Tuple[int, int], int]``
            A mapping from ``(batch_index, action_index)`` to ``(batch_size * num_entities)``,
            representing which action indices correspond to which entity indices in the returned
            ``flattened_linking_scores`` tensor.
        """
        batch_size, num_entities, num_question_tokens = linking_scores.size()
        entity_map: Dict[Tuple[int, str], int] = {}
        for batch_index, world in enumerate(worlds):
            for entity_index, entity in enumerate(world.table_graph.entities):
                entity_map[(batch_index, entity)] = batch_index * num_entities + entity_index
        actions_to_entities: Dict[Tuple[int, int], int] = {}
        for batch_index, action_list in enumerate(actions):
            for action_index, action in enumerate(action_list):
                if not action[0]:
                    # This action is padding.
                    continue
                _, production = action[0].split(' -> ')
                entity_index = entity_map.get((batch_index, production), None)
                if entity_index is not None:
                    actions_to_entities[(batch_index, action_index)] = entity_index
        flattened_linking_scores = linking_scores.view(batch_size * num_entities, num_question_tokens)
        return flattened_linking_scores, actions_to_entities

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions.  This is (confusingly) a separate notion from the "decoder"
        in "encoder/decoder", where that decoder logic lives in ``WikiTablesDecoderStep``.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        action_mapping = output_dict['action_mapping']
        best_actions = output_dict["best_action_sequence"]
        debug_infos = output_dict['debug_info']
        batch_action_info = []
        for batch_index, (predicted_actions, debug_info) in enumerate(zip(best_actions, debug_infos)):
            instance_action_info = []
            for predicted_action, action_debug_info in zip(predicted_actions, debug_info):
                action_info = {}
                action_info['predicted_action'] = predicted_action
                considered_actions = action_debug_info['considered_actions']
                probabilities = action_debug_info['probabilities']
                actions = []
                for action, probability in zip(considered_actions, probabilities):
                    if action != -1:
                        actions.append((action_mapping[(batch_index, action)], probability))
                actions.sort()
                considered_actions, probabilities = zip(*actions)
                action_info['considered_actions'] = considered_actions
                action_info['action_probabilities'] = probabilities
                action_info['question_attention'] = action_debug_info.get('question_attention', [])
                instance_action_info.append(action_info)
            batch_action_info.append(instance_action_info)
        output_dict["predicted_actions"] = batch_action_info
        return output_dict
