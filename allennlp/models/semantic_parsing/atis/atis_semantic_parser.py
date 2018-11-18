import logging
from typing import Any, Dict, List, Tuple

import difflib
import sqlparse
from overrides import overrides
import torch

from allennlp.common.util import pad_sequence_to_length
from allennlp.data import Vocabulary
from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.semparse.executors import SqlExecutor
from allennlp.models.model import Model
from allennlp.modules import Attention, Seq2SeqEncoder, TextFieldEmbedder, Embedding
from allennlp.nn import util
from allennlp.semparse.worlds import AtisWorld
from allennlp.semparse.contexts.atis_sql_table_context import NUMERIC_NONTERMINALS
from allennlp.semparse.contexts.sql_context_utils import action_sequence_to_sql
from allennlp.state_machines.states import GrammarBasedState
from allennlp.state_machines.transition_functions.linking_transition_function import LinkingTransitionFunction
from allennlp.state_machines import BeamSearch
from allennlp.state_machines.trainers import MaximumMarginalLikelihood
from allennlp.state_machines.states import GrammarStatelet, RnnStatelet
from allennlp.training.metrics import Average

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("atis_parser")
class AtisSemanticParser(Model):
    """
    Parameters
    ----------
    vocab : ``Vocabulary``
    utterance_embedder : ``TextFieldEmbedder``
        Embedder for utterances.
    action_embedding_dim : ``int``
        Dimension to use for action embeddings.
    encoder : ``Seq2SeqEncoder``
        The encoder to use for the input utterance.
    decoder_beam_search : ``BeamSearch``
        Beam search used to retrieve best sequences after training.
    max_decoding_steps : ``int``
        When we're decoding with a beam search, what's the maximum number of steps we should take?
        This only applies at evaluation time, not during training.
    input_attention: ``Attention``
        We compute an attention over the input utterance at each step of the decoder, using the
        decoder hidden state as the query.  Passed to the transition function.
    add_action_bias : ``bool``, optional (default=True)
        If ``True``, we will learn a bias weight for each action that gets used when predicting
        that action, in addition to its embedding.
    dropout : ``float``, optional (default=0)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer).
    rule_namespace : ``str``, optional (default=rule_labels)
        The vocabulary namespace to use for production rules.  The default corresponds to the
        default used in the dataset reader, so you likely don't need to modify this.
    database_file: ``str``, optional (default=/atis/atis.db)
        The path of the SQLite database when evaluating SQL queries. SQLite is disk based, so we need
        the file location to connect to it.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 utterance_embedder: TextFieldEmbedder,
                 action_embedding_dim: int,
                 encoder: Seq2SeqEncoder,
                 decoder_beam_search: BeamSearch,
                 max_decoding_steps: int,
                 input_attention: Attention,
                 add_action_bias: bool = True,
                 training_beam_size: int = None,
                 decoder_num_layers: int = 1,
                 dropout: float = 0.0,
                 rule_namespace: str = 'rule_labels',
                 database_file='/atis/atis.db') -> None:
        # Atis semantic parser init
        super().__init__(vocab)
        self._utterance_embedder = utterance_embedder
        self._encoder = encoder
        self._max_decoding_steps = max_decoding_steps
        self._add_action_bias = add_action_bias
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._rule_namespace = rule_namespace
        self._exact_match = Average()
        self._valid_sql_query = Average()
        self._action_similarity = Average()
        self._denotation_accuracy = Average()

        self._executor = SqlExecutor(database_file)
        self._action_padding_index = -1  # the padding value used by IndexField
        num_actions = vocab.get_vocab_size(self._rule_namespace)
        if self._add_action_bias:
            input_action_dim = action_embedding_dim + 1
        else:
            input_action_dim = action_embedding_dim
        self._action_embedder = Embedding(num_embeddings=num_actions, embedding_dim=input_action_dim)
        self._output_action_embedder = Embedding(num_embeddings=num_actions, embedding_dim=action_embedding_dim)


        # This is what we pass as input in the first step of decoding, when we don't have a
        # previous action, or a previous utterance attention.
        self._first_action_embedding = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))
        self._first_attended_utterance = torch.nn.Parameter(torch.FloatTensor(encoder.get_output_dim()))
        torch.nn.init.normal_(self._first_action_embedding)
        torch.nn.init.normal_(self._first_attended_utterance)

        self._num_entity_types = 2  # TODO(kevin): get this in a more principled way somehow?
        self._entity_type_decoder_embedding = Embedding(self._num_entity_types, action_embedding_dim)
        self._decoder_num_layers = decoder_num_layers

        self._beam_search = decoder_beam_search
        self._decoder_trainer = MaximumMarginalLikelihood(training_beam_size)
        self._transition_function = LinkingTransitionFunction(encoder_output_dim=self._encoder.get_output_dim(),
                                                              action_embedding_dim=action_embedding_dim,
                                                              input_attention=input_attention,
                                                              predict_start_type_separately=False,
                                                              add_action_bias=self._add_action_bias,
                                                              dropout=dropout,
                                                              num_layers=self._decoder_num_layers)

    @overrides
    def forward(self,  # type: ignore
                utterance: Dict[str, torch.LongTensor],
                world: List[AtisWorld],
                actions: List[List[ProductionRule]],
                linking_scores: torch.Tensor,
                target_action_sequence: torch.LongTensor = None,
                sql_queries: List[List[str]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        We set up the initial state for the decoder, and pass that state off to either a DecoderTrainer,
        if we're training, or a BeamSearch for inference, if we're not.

        Parameters
        ----------
        utterance : Dict[str, torch.LongTensor]
            The output of ``TextField.as_array()`` applied on the utterance ``TextField``. This will
            be passed through a ``TextFieldEmbedder`` and then through an encoder.
        world : ``List[AtisWorld]``
            We use a ``MetadataField`` to get the ``World`` for each input instance.  Because of
            how ``MetadataField`` works, this gets passed to us as a ``List[AtisWorld]``,
        actions : ``List[List[ProductionRule]]``
            A list of all possible actions for each ``World`` in the batch, indexed into a
            ``ProductionRule`` using a ``ProductionRuleField``.  We will embed all of these
            and use the embeddings to determine which action to take at each timestep in the
            decoder.
        linking_scores: ``torch.Tensor``
            A matrix of the linking the utterance tokens and the entities. This is a binary matrix that
            is deterministically generated where each entry indicates whether a token generated an entity.
            This tensor has shape ``(batch_size, num_entities, num_utterance_tokens)``.
        target_action_sequence : torch.Tensor, optional (default=None)
            The action sequence for the correct action sequence, where each action is an index into the list
            of possible actions.  This tensor has shape ``(batch_size, sequence_length, 1)``. We remove the
            trailing dimension.
        sql_queries : List[List[str]], optional (default=None)
            A list of the SQL queries that are given during training or validation.
        """
        initial_state = self._get_initial_state(utterance, world, actions, linking_scores)
        batch_size = linking_scores.shape[0]
        if target_action_sequence is not None:
            # Remove the trailing dimension (from ListField[ListField[IndexField]]).
            target_action_sequence = target_action_sequence.squeeze(-1)
            target_mask = target_action_sequence != self._action_padding_index
        else:
            target_mask = None

        if self.training:
            # target_action_sequence is of shape (batch_size, 1, sequence_length) here after we unsqueeze it for
            # the MML trainer.
            return self._decoder_trainer.decode(initial_state,
                                                self._transition_function,
                                                (target_action_sequence.unsqueeze(1), target_mask.unsqueeze(1)))
        else:
            # TODO(kevin) Move some of this functionality to a separate method for computing validation outputs.
            action_mapping = {}
            for batch_index, batch_actions in enumerate(actions):
                for action_index, action in enumerate(batch_actions):
                    action_mapping[(batch_index, action_index)] = action[0]
            outputs: Dict[str, Any] = {'action_mapping': action_mapping}
            outputs['linking_scores'] = linking_scores
            if target_action_sequence is not None:
                outputs['loss'] = self._decoder_trainer.decode(initial_state,
                                                               self._transition_function,
                                                               (target_action_sequence.unsqueeze(1),
                                                                target_mask.unsqueeze(1)))['loss']
            num_steps = self._max_decoding_steps
            # This tells the state to start keeping track of debug info, which we'll pass along in
            # our output dictionary.
            initial_state.debug_info = [[] for _ in range(batch_size)]
            best_final_states = self._beam_search.search(num_steps,
                                                         initial_state,
                                                         self._transition_function,
                                                         keep_final_unfinished_states=False)
            outputs['best_action_sequence'] = []
            outputs['debug_info'] = []
            outputs['entities'] = []
            outputs['predicted_sql_query'] = []
            outputs['sql_queries'] = []
            outputs['utterance'] = []
            outputs['tokenized_utterance'] = []

            for i in range(batch_size):
                # Decoding may not have terminated with any completed valid SQL queries, if `num_steps`
                # isn't long enough (or if the model is not trained enough and gets into an
                # infinite action loop).
                if i not in best_final_states:
                    self._exact_match(0)
                    self._denotation_accuracy(0)
                    self._valid_sql_query(0)
                    self._action_similarity(0)
                    outputs['predicted_sql_query'].append('')
                    continue

                best_action_indices = best_final_states[i][0].action_history[0]

                action_strings = [action_mapping[(i, action_index)]
                                  for action_index in best_action_indices]
                predicted_sql_query = action_sequence_to_sql(action_strings)

                if target_action_sequence is not None:
                    # Use a Tensor, not a Variable, to avoid a memory leak.
                    targets = target_action_sequence[i].data
                    sequence_in_targets = 0
                    sequence_in_targets = self._action_history_match(best_action_indices, targets)
                    self._exact_match(sequence_in_targets)

                    similarity = difflib.SequenceMatcher(None, best_action_indices, targets)
                    self._action_similarity(similarity.ratio())

                if sql_queries and sql_queries[i]:
                    denotation_correct = self._executor.evaluate_sql_query(predicted_sql_query, sql_queries[i])
                    self._denotation_accuracy(denotation_correct)
                    outputs['sql_queries'].append(sql_queries[i])

                outputs['utterance'].append(world[i].utterances[-1])
                outputs['tokenized_utterance'].append([token.text
                                                       for token in world[i].tokenized_utterances[-1]])
                outputs['entities'].append(world[i].entities)
                outputs['best_action_sequence'].append(action_strings)
                outputs['predicted_sql_query'].append(sqlparse.format(predicted_sql_query, reindent=True))
                outputs['debug_info'].append(best_final_states[i][0].debug_info[0])  # type: ignore
            return outputs

    def _get_initial_state(self,
                           utterance: Dict[str, torch.LongTensor],
                           worlds: List[AtisWorld],
                           actions: List[List[ProductionRule]],
                           linking_scores: torch.Tensor) -> GrammarBasedState:
        embedded_utterance = self._utterance_embedder(utterance)
        utterance_mask = util.get_text_field_mask(utterance).float()

        batch_size = embedded_utterance.size(0)
        num_entities = max([len(world.entities) for world in worlds])

        # entity_types: tensor with shape (batch_size, num_entities)
        entity_types, _ = self._get_type_vector(worlds, num_entities, embedded_utterance)

        # (batch_size, num_utterance_tokens, embedding_dim)
        encoder_input = embedded_utterance

        # (batch_size, utterance_length, encoder_output_dim)
        encoder_outputs = self._dropout(self._encoder(encoder_input, utterance_mask))

        # This will be our initial hidden state and memory cell for the decoder LSTM.
        final_encoder_output = util.get_final_encoder_states(encoder_outputs,
                                                             utterance_mask,
                                                             self._encoder.is_bidirectional())
        memory_cell = encoder_outputs.new_zeros(batch_size, self._encoder.get_output_dim())
        initial_score = embedded_utterance.data.new_zeros(batch_size)

        # To make grouping states together in the decoder easier, we convert the batch dimension in
        # all of our tensors into an outer list.  For instance, the encoder outputs have shape
        # `(batch_size, utterance_length, encoder_output_dim)`.  We need to convert this into a list
        # of `batch_size` tensors, each of shape `(utterance_length, encoder_output_dim)`.  Then we
        # won't have to do any index selects, or anything, we'll just do some `torch.cat()`s.
        initial_score_list = [initial_score[i] for i in range(batch_size)]
        encoder_output_list = [encoder_outputs[i] for i in range(batch_size)]
        utterance_mask_list = [utterance_mask[i] for i in range(batch_size)]
        initial_rnn_state = []
        for i in range(batch_size):
            if self._decoder_num_layers > 1:
                initial_rnn_state.append(RnnStatelet(final_encoder_output[i].repeat(self._decoder_num_layers, 1),
                                                     memory_cell[i].repeat(self._decoder_num_layers, 1),
                                                     self._first_action_embedding,
                                                     self._first_attended_utterance,
                                                     encoder_output_list,
                                                     utterance_mask_list))
            else:
                initial_rnn_state.append(RnnStatelet(final_encoder_output[i],
                                                     memory_cell[i],
                                                     self._first_action_embedding,
                                                     self._first_attended_utterance,
                                                     encoder_output_list,
                                                     utterance_mask_list))


        initial_grammar_state = [self._create_grammar_state(worlds[i],
                                                            actions[i],
                                                            linking_scores[i],
                                                            entity_types[i])
                                 for i in range(batch_size)]

        initial_state = GrammarBasedState(batch_indices=list(range(batch_size)),
                                          action_history=[[] for _ in range(batch_size)],
                                          score=initial_score_list,
                                          rnn_state=initial_rnn_state,
                                          grammar_state=initial_grammar_state,
                                          possible_actions=actions,
                                          debug_info=None)
        return initial_state

    @staticmethod
    def _get_type_vector(worlds: List[AtisWorld],
                         num_entities: int,
                         tensor: torch.Tensor = None) -> Tuple[torch.LongTensor, Dict[int, int]]:
        """
        Produces the encoding for each entity's type. In addition, a map from a flattened entity
        index to type is returned to combine entity type operations into one method.

        Parameters
        ----------
        worlds : ``List[AtisWorld]``
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
            entities = [('number', entity)
                        if any([entity.startswith(numeric_nonterminal)
                                for numeric_nonterminal in NUMERIC_NONTERMINALS])
                        else ('string', entity)
                        for entity in world.entities]

            for entity_index, entity in enumerate(entities):
                # We need numbers to be first, then strings, since our entities are going to be
                # sorted. We do a split by type and then a merge later, and it relies on this sorting.
                if entity[0] == 'number':
                    entity_type = 1
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

    @staticmethod
    def _action_history_match(predicted: List[int], targets: torch.LongTensor) -> int:
        # TODO(mattg): this could probably be moved into a FullSequenceMatch metric, or something.
        # Check if target is big enough to cover prediction (including start/end symbols)
        if len(predicted) > targets.size(0):
            return 0
        predicted_tensor = targets.new_tensor(predicted)
        targets_trimmed = targets[:len(predicted)]
        # Return 1 if the predicted sequence is anywhere in the list of targets.
        return predicted_tensor.equal(targets_trimmed)

    @staticmethod
    def is_nonterminal(token: str):
        if token[0] == '"' and token[-1] == '"':
            return False
        return True

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        We track four metrics here:

            1. exact_match, which is the percentage of the time that our best output action sequence
            matches the SQL query exactly.

            2. denotation_acc, which is the percentage of examples where we get the correct
            denotation.  This is the typical "accuracy" metric, and it is what you should usually
            report in an experimental result.  You need to be careful, though, that you're
            computing this on the full data, and not just the subset that can be parsed. (make sure
            you pass "keep_if_unparseable=True" to the dataset reader, which we do for validation data,
            but not training data).

            3. valid_sql_query, which is the percentage of time that decoding actually produces a
            valid SQL query.  We might not produce a valid SQL query if the decoder gets
            into a repetitive loop, or we're trying to produce a super long SQL query and run
            out of time steps, or something.

            4. action_similarity, which is how similar the action sequence predicted is to the actual
            action sequence. This is basically a soft measure of exact_match.
        """
        return {
                'exact_match': self._exact_match.get_metric(reset),
                'denotation_acc': self._denotation_accuracy.get_metric(reset),
                'valid_sql_query': self._valid_sql_query.get_metric(reset),
                'action_similarity': self._action_similarity.get_metric(reset)
                }

    def _create_grammar_state(self,
                              world: AtisWorld,
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
        world : ``AtisWorld``
            From the input to ``forward`` for a single batch instance.
        possible_actions : ``List[ProductionRule]``
            From the input to ``forward`` for a single batch instance.
        linking_scores : ``torch.Tensor``
            Assumed to have shape ``(num_entities, num_utterance_tokens)`` (i.e., there is no batch
            dimension).
        entity_types : ``torch.Tensor``
            Assumed to have shape ``(num_entities,)`` (i.e., there is no batch dimension).
        """
        action_map = {}
        for action_index, action in enumerate(possible_actions):
            action_string = action[0]
            action_map[action_string] = action_index

        valid_actions = world.valid_actions
        entity_map = {}
        entities = world.entities

        for entity_index, entity in enumerate(entities):
            entity_map[entity] = entity_index

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

            if global_actions:
                global_action_tensors, global_action_ids = zip(*global_actions)
                global_action_tensor = entity_types.new_tensor(torch.cat(global_action_tensors, dim=0),
                                                               dtype=torch.long)
                global_input_embeddings = self._action_embedder(global_action_tensor)
                global_output_embeddings = self._output_action_embedder(global_action_tensor)
                translated_valid_actions[key]['global'] = (global_input_embeddings,
                                                           global_output_embeddings,
                                                           list(global_action_ids))
            if linked_actions:
                linked_rules, linked_action_ids = zip(*linked_actions)
                entities = linked_rules
                entity_ids = [entity_map[entity] for entity in entities]
                entity_linking_scores = linking_scores[entity_ids]
                entity_type_tensor = entity_types[entity_ids]
                entity_type_embeddings = self._entity_type_decoder_embedding(entity_type_tensor)
                entity_type_embeddings = entity_types.new_tensor(entity_type_embeddings, dtype=torch.float)
                translated_valid_actions[key]['linked'] = (entity_linking_scores,
                                                           entity_type_embeddings,
                                                           list(linked_action_ids))

        return GrammarStatelet(['statement'],
                               translated_valid_actions,
                               self.is_nonterminal)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions.  This is (confusingly) a separate notion from the "decoder"
        in "encoder/decoder", where that decoder logic lives in ``TransitionFunction``.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_actions`` to the ``output_dict``.
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
                action_info['utterance_attention'] = action_debug_info.get('question_attention', [])
                instance_action_info.append(action_info)
            batch_action_info.append(instance_action_info)
        output_dict["predicted_actions"] = batch_action_info
        return output_dict
