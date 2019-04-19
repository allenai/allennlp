from typing import Any, Dict, List, Tuple

import math
from overrides import overrides
import torch

from allennlp.common.util import pad_sequence_to_length
from allennlp.data import Vocabulary
from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.models.model import Model
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder, FeedForward, Embedding
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.nn import util
from allennlp.semparse.type_declarations import type_declaration
from allennlp.semparse.type_declarations.type_declaration import START_SYMBOL
from allennlp.semparse.worlds.quarel_world import QuarelWorld
from allennlp.semparse import ParsingError
from allennlp.state_machines import BeamSearch
from allennlp.state_machines.states import GrammarBasedState, GrammarStatelet, RnnStatelet
from allennlp.state_machines.trainers import MaximumMarginalLikelihood
from allennlp.state_machines.transition_functions import LinkingTransitionFunction
from allennlp.training.metrics import Average, CategoricalAccuracy


@Model.register("quarel_parser")
class QuarelSemanticParser(Model):
    """
    A ``QuarelSemanticParser`` is a variant of ``WikiTablesSemanticParser`` with various
    tweaks and changes.

    Parameters
    ----------
    vocab : ``Vocabulary``
    question_embedder : ``TextFieldEmbedder``
        Embedder for questions.
    action_embedding_dim : ``int``
        Dimension to use for action embeddings.
    encoder : ``Seq2SeqEncoder``
        The encoder to use for the input question.
    decoder_beam_search : ``BeamSearch``
        When we're not training, this is how we will do decoding.
    max_decoding_steps : ``int``
        When we're decoding with a beam search, what's the maximum number of steps we should take?
        This only applies at evaluation time, not during training.
    attention : ``Attention``
        We compute an attention over the input question at each step of the decoder, using the
        decoder hidden state as the query.  Passed to the transition function.
    dropout : ``float``, optional (default=0)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer).
    num_linking_features : ``int``, optional (default=10)
        We need to construct a parameter vector for the linking features, so we need to know how
        many there are.  The default of 8 here matches the default in the ``KnowledgeGraphField``,
        which is to use all eight defined features. If this is 0, another term will be added to the
        linking score. This term contains the maximum similarity value from the entity's neighbors
        and the question.
    use_entities : ``bool``, optional (default=False)
        Whether dynamic entities are part of the action space
    num_entity_bits : ``int``, optional (default=0)
        Whether any bits are added to encoder input/output to represent tagged entities
    entity_bits_output : ``bool``, optional (default=False)
        Whether entity bits are added to the encoder output or input
    denotation_only : ``bool``, optional (default=False)
        Whether to only predict target denotation, skipping the the whole logical form decoder
    entity_similarity_mode : ``str``, optional (default="dot_product")
        How to compute vector similarity between question and entity tokens, can take values
        "dot_product" or "weighted_dot_product" (learned weights on each dimension)
    rule_namespace : ``str``, optional (default=rule_labels)
        The vocabulary namespace to use for production rules.  The default corresponds to the
        default used in the dataset reader, so you likely don't need to modify this.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 question_embedder: TextFieldEmbedder,
                 action_embedding_dim: int,
                 encoder: Seq2SeqEncoder,
                 decoder_beam_search: BeamSearch,
                 max_decoding_steps: int,
                 attention: Attention,
                 mixture_feedforward: FeedForward = None,
                 add_action_bias: bool = True,
                 dropout: float = 0.0,
                 num_linking_features: int = 0,
                 num_entity_bits: int = 0,
                 entity_bits_output: bool = True,
                 use_entities: bool = False,
                 denotation_only: bool = False,
                 # Deprecated parameter to load older models
                 entity_encoder: Seq2VecEncoder = None,  # pylint: disable=unused-argument
                 entity_similarity_mode: str = "dot_product",
                 rule_namespace: str = 'rule_labels') -> None:
        super(QuarelSemanticParser, self).__init__(vocab)
        self._question_embedder = question_embedder
        self._encoder = encoder
        self._beam_search = decoder_beam_search
        self._max_decoding_steps = max_decoding_steps
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._rule_namespace = rule_namespace
        self._denotation_accuracy = Average()
        self._action_sequence_accuracy = Average()
        self._has_logical_form = Average()

        self._embedding_dim = question_embedder.get_output_dim()
        self._use_entities = use_entities

        # Note: there's only one non-trivial entity type in QuaRel for now, so most of the
        # entity_type stuff is irrelevant
        self._num_entity_types = 4  # TODO(mattg): get this in a more principled way somehow?
        self._num_start_types = 1 # Hardcoded until we feed lf syntax into the model
        self._entity_type_encoder_embedding = Embedding(self._num_entity_types, self._embedding_dim)
        self._entity_type_decoder_embedding = Embedding(self._num_entity_types, action_embedding_dim)

        self._entity_similarity_layer = None
        self._entity_similarity_mode = entity_similarity_mode
        if self._entity_similarity_mode == "weighted_dot_product":
            self._entity_similarity_layer = \
                TimeDistributed(torch.nn.Linear(self._embedding_dim, 1, bias=False))
            # Center initial values around unweighted dot product
            self._entity_similarity_layer._module.weight.data += 1  # pylint: disable=protected-access
        elif self._entity_similarity_mode == "dot_product":
            pass
        else:
            raise ValueError("Invalid entity_similarity_mode: {}".format(self._entity_similarity_mode))

        if num_linking_features > 0:
            self._linking_params = torch.nn.Linear(num_linking_features, 1)
        else:
            self._linking_params = None

        self._decoder_trainer = MaximumMarginalLikelihood()

        self._encoder_output_dim = self._encoder.get_output_dim()
        if entity_bits_output:
            self._encoder_output_dim += num_entity_bits

        self._entity_bits_output = entity_bits_output

        self._debug_count = 10

        self._num_denotation_cats = 2  # Hardcoded for simplicity
        self._denotation_only = denotation_only
        if self._denotation_only:
            self._denotation_accuracy_cat = CategoricalAccuracy()
            self._denotation_classifier = torch.nn.Linear(self._encoder_output_dim,
                                                          self._num_denotation_cats)
            # Rest of init not needed for denotation only where no decoding to actions needed
            return

        self._action_padding_index = -1  # the padding value used by IndexField
        num_actions = vocab.get_vocab_size(self._rule_namespace)
        self._num_actions = num_actions
        self._action_embedder = Embedding(num_embeddings=num_actions, embedding_dim=action_embedding_dim)
        # We are tying the action embeddings used for input and output
        # self._output_action_embedder = Embedding(num_embeddings=num_actions, embedding_dim=action_embedding_dim)
        self._output_action_embedder = self._action_embedder  # tied weights
        self._add_action_bias = add_action_bias
        if self._add_action_bias:
            self._action_biases = Embedding(num_embeddings=num_actions, embedding_dim=1)

        # This is what we pass as input in the first step of decoding, when we don't have a
        # previous action, or a previous question attention.
        self._first_action_embedding = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))
        self._first_attended_question = torch.nn.Parameter(torch.FloatTensor(self._encoder_output_dim))
        torch.nn.init.normal_(self._first_action_embedding)
        torch.nn.init.normal_(self._first_attended_question)

        self._decoder_step = LinkingTransitionFunction(encoder_output_dim=self._encoder_output_dim,
                                                       action_embedding_dim=action_embedding_dim,
                                                       input_attention=attention,
                                                       num_start_types=self._num_start_types,
                                                       predict_start_type_separately=False,
                                                       add_action_bias=self._add_action_bias,
                                                       mixture_feedforward=mixture_feedforward,
                                                       dropout=dropout)

    @overrides
    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                table: Dict[str, torch.LongTensor],
                world: List[QuarelWorld],
                actions: List[List[ProductionRule]],
                entity_bits: torch.Tensor = None,
                denotation_target: torch.Tensor = None,
                target_action_sequences: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        """
        In this method we encode the table entities, link them to words in the question, then
        encode the question. Then we set up the initial state for the decoder, and pass that
        state off to either a DecoderTrainer, if we're training, or a BeamSearch for inference,
        if we're not.

        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
           The output of ``TextField.as_array()`` applied on the question ``TextField``. This will
           be passed through a ``TextFieldEmbedder`` and then through an encoder.
        table : ``Dict[str, torch.LongTensor]``
            The output of ``KnowledgeGraphField.as_array()`` applied on the table
            ``KnowledgeGraphField``.  This output is similar to a ``TextField`` output, where each
            entity in the table is treated as a "token", and we will use a ``TextFieldEmbedder`` to
            get embeddings for each entity.
        world : ``List[QuarelWorld]``
            We use a ``MetadataField`` to get the ``World`` for each input instance.  Because of
            how ``MetadataField`` works, this gets passed to us as a ``List[QuarelWorld]``,
        actions : ``List[List[ProductionRule]]``
            A list of all possible actions for each ``World`` in the batch, indexed into a
            ``ProductionRule`` using a ``ProductionRuleField``.  We will embed all of these
            and use the embeddings to determine which action to take at each timestep in the
            decoder.
        entity_bits : ``torch.Tensor``, optional (default=None)
            Tensor encoding bits for the world entities.
        denotation_target : ``torch.Tensor``, optional (default=None)
            If model's field ``denotation_only`` is True, this is the tensor target denotation.
        target_action_sequences : torch.Tensor, optional (default=None)
           A list of possibly valid action sequences, where each action is an index into the list
           of possible actions.  This tensor has shape ``(batch_size, num_action_sequences,
           sequence_length)``.
        metadata : List[Dict[str, Any]], optional (default=None).
            A dictionary of metadata for each batch element which has keys:
                question_tokens : ``List[str]``, optional.
                    The original string tokens in the question.
                world_extractions : ``nltk.Tree``, optional.
                    Extracted worlds from the question.
                answer_index : ``List[str]``, optional.
                    Index of the correct answer.
        """

        table_text = table['text']

        self._debug_count -= 1

        # (batch_size, question_length, embedding_dim)
        embedded_question = self._question_embedder(question)
        question_mask = util.get_text_field_mask(question).float()
        num_question_tokens = embedded_question.size(1)

        # (batch_size, num_entities, num_entity_tokens, embedding_dim)
        embedded_table = self._question_embedder(table_text, num_wrapping_dims=1)

        batch_size, num_entities, num_entity_tokens, _ = embedded_table.size()

        # entity_types: one-hot tensor with shape (batch_size, num_entities, num_types)
        # entity_type_dict: Dict[int, int], mapping flattened_entity_index -> type_index
        # These encode the same information, but for efficiency reasons later it's nice
        # to have one version as a tensor and one that's accessible on the cpu.
        entity_types, entity_type_dict = self._get_type_vector(world, num_entities, embedded_table)

        if self._use_entities:

            if self._entity_similarity_mode == "dot_product":
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

                linking_scores = question_entity_similarity_max_score
            elif self._entity_similarity_mode == "weighted_dot_product":
                embedded_table = embedded_table / (embedded_table.norm(dim=-1, keepdim=True) + 1e-13)
                embedded_question = embedded_question / (embedded_question.norm(dim=-1, keepdim=True) + 1e-13)
                eqe = embedded_question.unsqueeze(1).expand(-1, num_entities*num_entity_tokens, -1, -1)
                ete = embedded_table.view(batch_size, num_entities*num_entity_tokens, self._embedding_dim)
                ete = ete.unsqueeze(2).expand(-1, -1, num_question_tokens, -1)
                product = torch.mul(eqe, ete)
                product = product.view(batch_size,
                                       num_question_tokens*num_entities*num_entity_tokens,
                                       self._embedding_dim)
                question_entity_similarity = self._entity_similarity_layer(product)
                question_entity_similarity = question_entity_similarity.view(batch_size,
                                                                             num_entities,
                                                                             num_entity_tokens,
                                                                             num_question_tokens)

                # (batch_size, num_entities, num_question_tokens)
                question_entity_similarity_max_score, _ = torch.max(question_entity_similarity, 2)
                linking_scores = question_entity_similarity_max_score

            # (batch_size, num_entities, num_question_tokens, num_features)
            linking_features = table['linking']

            if self._linking_params is not None:
                feature_scores = self._linking_params(linking_features).squeeze(3)
                linking_scores = linking_scores + feature_scores

            # (batch_size, num_question_tokens, num_entities)
            linking_probabilities = self._get_linking_probabilities(world, linking_scores.transpose(1, 2),
                                                                    question_mask, entity_type_dict)
            encoder_input = embedded_question
        else:
            if entity_bits is not None and not self._entity_bits_output:
                encoder_input = torch.cat([embedded_question, entity_bits], 2)
            else:
                encoder_input = embedded_question

            # Fake linking_scores added for downstream code to not object
            linking_scores = question_mask.clone().fill_(0).unsqueeze(1)
            linking_probabilities = None

        # (batch_size, question_length, encoder_output_dim)
        encoder_outputs = self._dropout(self._encoder(encoder_input, question_mask))

        if self._entity_bits_output and entity_bits is not None:
            encoder_outputs = torch.cat([encoder_outputs, entity_bits], 2)

        # This will be our initial hidden state and memory cell for the decoder LSTM.
        final_encoder_output = util.get_final_encoder_states(encoder_outputs,
                                                             question_mask,
                                                             self._encoder.is_bidirectional())
        # For predicting a categorical denotation directly
        if self._denotation_only:
            denotation_logits = self._denotation_classifier(final_encoder_output)
            loss = torch.nn.functional.cross_entropy(denotation_logits, denotation_target.view(-1))
            self._denotation_accuracy_cat(denotation_logits, denotation_target)
            return {"loss": loss}

        memory_cell = encoder_outputs.new_zeros(batch_size, self._encoder_output_dim)

        _, num_entities, num_question_tokens = linking_scores.size()

        if target_action_sequences is not None:
            # Remove the trailing dimension (from ListField[ListField[IndexField]]).
            target_action_sequences = target_action_sequences.squeeze(-1)
            target_mask = target_action_sequences != self._action_padding_index
        else:
            target_mask = None

        # To make grouping states together in the decoder easier, we convert the batch dimension in
        # all of our tensors into an outer list.  For instance, the encoder outputs have shape
        # `(batch_size, question_length, encoder_output_dim)`.  We need to convert this into a list
        # of `batch_size` tensors, each of shape `(question_length, encoder_output_dim)`.  Then we
        # won't have to do any index selects, or anything, we'll just do some `torch.cat()`s.
        encoder_output_list = [encoder_outputs[i] for i in range(batch_size)]
        question_mask_list = [question_mask[i] for i in range(batch_size)]
        initial_rnn_state = []

        for i in range(batch_size):
            initial_rnn_state.append(RnnStatelet(final_encoder_output[i],
                                                 memory_cell[i],
                                                 self._first_action_embedding,
                                                 self._first_attended_question,
                                                 encoder_output_list,
                                                 question_mask_list))

        initial_grammar_state = [self._create_grammar_state(world[i], actions[i],
                                                            linking_scores[i], entity_types[i])
                                 for i in range(batch_size)]

        initial_score = initial_rnn_state[0].hidden_state.new_zeros(batch_size)
        initial_score_list = [initial_score[i] for i in range(batch_size)]
        initial_state = GrammarBasedState(batch_indices=list(range(batch_size)),
                                          action_history=[[] for _ in range(batch_size)],
                                          score=initial_score_list,
                                          rnn_state=initial_rnn_state,
                                          grammar_state=initial_grammar_state,
                                          possible_actions=actions,
                                          extras=None,
                                          debug_info=None)

        if self.training:
            outputs = self._decoder_trainer.decode(initial_state,
                                                   self._decoder_step,
                                                   (target_action_sequences, target_mask))
            return outputs

        else:
            action_mapping = {}
            for batch_index, batch_actions in enumerate(actions):
                for action_index, action in enumerate(batch_actions):
                    action_mapping[(batch_index, action_index)] = action[0]
            outputs = {'action_mapping': action_mapping}
            if target_action_sequences is not None:
                outputs['loss'] = self._decoder_trainer.decode(initial_state,
                                                               self._decoder_step,
                                                               (target_action_sequences, target_mask))['loss']

            num_steps = self._max_decoding_steps
            # This tells the state to start keeping track of debug info, which we'll pass along in
            # our output dictionary.
            initial_state.debug_info = [[] for _ in range(batch_size)]
            best_final_states = self._beam_search.search(num_steps,
                                                         initial_state,
                                                         self._decoder_step,
                                                         keep_final_unfinished_states=False)
            outputs['best_action_sequence'] = []
            outputs['debug_info'] = []
            outputs['entities'] = []
            if self._linking_params is not None:
                outputs['linking_scores'] = linking_scores
                outputs['feature_scores'] = feature_scores
                outputs['linking_features'] = linking_features
            if self._use_entities:
                outputs['linking_probabilities'] = linking_probabilities
            if entity_bits is not None:
                outputs['entity_bits'] = entity_bits
            # outputs['similarity_scores'] = question_entity_similarity_max_score
            outputs['logical_form'] = []
            outputs['denotation_acc'] = []
            outputs['score'] = []
            outputs['parse_acc'] = []
            outputs['answer_index'] = []
            if metadata is not None:
                outputs['question_tokens'] = []
                outputs['world_extractions'] = []
            for i in range(batch_size):
                if metadata is not None:
                    outputs['question_tokens'].append(metadata[i].get('question_tokens', []))
                if metadata is not None:
                    outputs['world_extractions'].append(metadata[i].get('world_extractions', {}))
                outputs['entities'].append(world[i].table_graph.entities)
                # Decoding may not have terminated with any completed logical forms, if `num_steps`
                # isn't long enough (or if the model is not trained enough and gets into an
                # infinite action loop).
                if i in best_final_states:
                    best_action_indices = best_final_states[i][0].action_history[0]
                    sequence_in_targets = 0
                    if target_action_sequences is not None:
                        targets = target_action_sequences[i].data
                        sequence_in_targets = self._action_history_match(best_action_indices, targets)
                        self._action_sequence_accuracy(sequence_in_targets)
                    action_strings = [action_mapping[(i, action_index)] for action_index in best_action_indices]
                    try:
                        self._has_logical_form(1.0)
                        logical_form = world[i].get_logical_form(action_strings, add_var_function=False)
                    except ParsingError:
                        self._has_logical_form(0.0)
                        logical_form = 'Error producing logical form'
                    denotation_accuracy = 0.0
                    predicted_answer_index = world[i].execute(logical_form)
                    if metadata is not None and 'answer_index' in metadata[i]:
                        answer_index = metadata[i]['answer_index']
                        denotation_accuracy = self._denotation_match(predicted_answer_index, answer_index)
                        self._denotation_accuracy(denotation_accuracy)
                    score = math.exp(best_final_states[i][0].score[0].data.cpu().item())
                    outputs['answer_index'].append(predicted_answer_index)
                    outputs['score'].append(score)
                    outputs['parse_acc'].append(sequence_in_targets)
                    outputs['best_action_sequence'].append(action_strings)
                    outputs['logical_form'].append(logical_form)
                    outputs['denotation_acc'].append(denotation_accuracy)
                    outputs['debug_info'].append(best_final_states[i][0].debug_info[0])  # type: ignore
                else:
                    outputs['parse_acc'].append(0)
                    outputs['logical_form'].append('')
                    outputs['denotation_acc'].append(0)
                    outputs['score'].append(0)
                    outputs['answer_index'].append(-1)
                    outputs['best_action_sequence'].append([])
                    outputs['debug_info'].append([])
                    self._has_logical_form(0.0)
            return outputs

    @staticmethod
    def _get_type_vector(worlds: List[QuarelWorld],
                         num_entities: int,
                         tensor: torch.Tensor) -> Tuple[torch.LongTensor, Dict[int, int]]:
        """
        Produces a tensor with shape ``(batch_size, num_entities)`` that encodes each entity's
        type. In addition, a map from a flattened entity index to type is returned to combine
        entity type operations into one method.

        Parameters
        ----------
        worlds : ``List[WikiTablesWorld]``
        num_entities : ``int``
        tensor : ``torch.Tensor``
            Used for copying the constructed list onto the right device.

        Returns
        -------
        A ``torch.LongTensor`` with shape ``(batch_size, num_entities)``.
        entity_types : ``Dict[int, int]``
            This is a mapping from ((batch_index * num_entities) + entity_index) to entity type id.
        """
        entity_types = {}
        batch_types = []
        for batch_index, world in enumerate(worlds):
            types = []
            for entity_index, entity in enumerate(world.table_graph.entities):
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
                types.append(entity_type)

                # For easier lookups later, we're actually using a _flattened_ version
                # of (batch_index, entity_index) for the key, because this is how the
                # linking scores are stored.
                flattened_entity_index = batch_index * num_entities + entity_index
                entity_types[flattened_entity_index] = entity_type
            padded = pad_sequence_to_length(types, num_entities, lambda: 0)
            batch_types.append(padded)
        return tensor.new_tensor(batch_types, dtype=torch.long), entity_types

    def _get_linking_probabilities(self,
                                   worlds: List[QuarelWorld],
                                   linking_scores: torch.FloatTensor,
                                   question_mask: torch.LongTensor,
                                   entity_type_dict: Dict[int, int]) -> torch.FloatTensor:
        """
        Produces the probability of an entity given a question word and type. The logic below
        separates the entities by type since the softmax normalization term sums over entities
        of a single type.

        Parameters
        ----------
        worlds : ``List[QuarelWorld]``
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

    def _denotation_match(self, predicted_answer_index: int, target_answer_index: int) -> float:
        if predicted_answer_index < 0:
            # Logical form doesn't properly resolve, we do random guess with appropriate credit
            return 1.0/self._num_denotation_cats
        elif predicted_answer_index == target_answer_index:
            return 1.0
        return 0.0

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        We track three metrics here:

            1. parse_acc, which is the percentage of the time that our best output action sequence
            corresponds to a correct logical form

            2. denotation_acc, which is the percentage of examples where we get the correct
            denotation, including spurious correct answers using the wrong logical form

            3. lf_percent, which is the percentage of time that decoding actually produces a
            finished logical form.  We might not produce a valid logical form if the decoder gets
            into a repetitive loop, or we're trying to produce a super long logical form and run
            out of time steps, or something.
        """
        if self._denotation_only:
            metrics = {'denotation_acc': self._denotation_accuracy_cat.get_metric(reset)}
        else:
            metrics = {
                    'parse_acc': self._action_sequence_accuracy.get_metric(reset),
                    'denotation_acc': self._denotation_accuracy.get_metric(reset),
                    'lf_percent': self._has_logical_form.get_metric(reset),
            }
        return metrics

    def _create_grammar_state(self,
                              world: QuarelWorld,
                              possible_actions: List[ProductionRule],
                              linking_scores: torch.Tensor,
                              entity_types: torch.Tensor) -> GrammarStatelet:
        """
        This method creates the GrammarStatelet object that's used for decoding.  Part of creating
        that is creating the `valid_actions` dictionary, which contains embedded representations of
        all of the valid actions.  So, we create that here as well.

        The inputs to this method are for a `single instance in the batch`; none of the tensors we
        create here are batched.  We grab the global action ids from the input
        ``ProductionRules``, and we use those to embed the valid actions for every
        non-terminal type.  We use the input ``linking_scores`` for non-global actions.

        Parameters
        ----------
        world : ``QuarelWorld``
            From the input to ``forward`` for a single batch instance.
        possible_actions : ``List[ProductionRule]``
            From the input to ``forward`` for a single batch instance.
        linking_scores : ``torch.Tensor``
            Assumed to have shape ``(num_entities, num_question_tokens)`` (i.e., there is no batch
            dimension).
        entity_types : ``torch.Tensor``
            Assumed to have shape ``(num_entities,)`` (i.e., there is no batch dimension).
        """
        action_map = {}
        for action_index, action in enumerate(possible_actions):
            action_string = action[0]
            action_map[action_string] = action_index
        entity_map = {}
        for entity_index, entity in enumerate(world.table_graph.entities):
            entity_map[entity] = entity_index

        valid_actions = world.get_valid_actions()
        translated_valid_actions: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor, List[int]]]] = {}
        for key, action_strings in valid_actions.items():
            translated_valid_actions[key] = {}
            # `key` here is a non-terminal from the grammar, and `action_strings` are all the valid
            # productions of that non-terminal.  We'll first split those productions by global vs.
            # linked action.
            action_indices = [action_map[action_string] for action_string in action_strings]
            production_rule_arrays = [(possible_actions[index], index) for index in action_indices]
            global_actions = []
            linked_actions = []
            for production_rule_array, action_index in production_rule_arrays:
                if production_rule_array[1]:
                    global_actions.append((production_rule_array[2], action_index))
                else:
                    linked_actions.append((production_rule_array[0], action_index))

            # Then we get the embedded representations of the global actions.
            global_action_tensors, global_action_ids = zip(*global_actions)
            global_action_tensor = torch.cat(global_action_tensors, dim=0)
            global_input_embeddings = self._action_embedder(global_action_tensor)
            if self._add_action_bias:
                global_action_biases = self._action_biases(global_action_tensor)
                global_input_embeddings = torch.cat([global_input_embeddings, global_action_biases], dim=-1)
            global_output_embeddings = self._output_action_embedder(global_action_tensor)
            translated_valid_actions[key]['global'] = (global_input_embeddings,
                                                       global_output_embeddings,
                                                       list(global_action_ids))

            # Then the representations of the linked actions.
            if linked_actions:
                linked_rules, linked_action_ids = zip(*linked_actions)
                entities = [rule.split(' -> ')[1] for rule in linked_rules]
                entity_ids = [entity_map[entity] for entity in entities]
                # (num_linked_actions, num_question_tokens)
                entity_linking_scores = linking_scores[entity_ids]
                # (num_linked_actions,)
                entity_type_tensor = entity_types[entity_ids]
                # (num_linked_actions, entity_type_embedding_dim)
                entity_type_embeddings = self._entity_type_decoder_embedding(entity_type_tensor)
                translated_valid_actions[key]['linked'] = (entity_linking_scores,
                                                           entity_type_embeddings,
                                                           list(linked_action_ids))

        return GrammarStatelet([START_SYMBOL],
                               translated_valid_actions,
                               type_declaration.is_nonterminal)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions.  This is (confusingly) a separate notion from the "decoder"
        in "encoder/decoder", where that decoder logic lives in ``FrictionQDecoderStep``.

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
