from typing import List, Set, Dict, Tuple
from collections import defaultdict

from overrides import overrides

import torch
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.seq2seq import START_SYMBOL, END_SYMBOL
from allennlp.data.semparse.type_declarations import type_declaration as types
from allennlp.data.semparse.type_declarations import nlvr_type_declaration as nlvr_types
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.nn.decoding import DecoderTrainer, BeamSearch
from allennlp.nn import util as nn_util
from allennlp.models.model import Model
from allennlp.models.encoder_decoders.wikitables_semantic_parser import WikiTablesDecoderStep
from allennlp.models.encoder_decoders.wikitables_semantic_parser import WikiTablesDecoderState


@Model.register("nlvr_parser")
class NlvrSemanticParser(Model):
    """
    ``NlvrSemanticParser`` is a semantic parsing model built for the NLVR domain.
    There is a lot of overlap with ``WikiTablesSemanticParser`` here. We may want to eventually move the
    common functionality into a more general transition-based parsing class.

    The main differences between this parser and what we have for Wikitables are that we have an agenda
    of actions instead of complete target action sequences, and accordingly the score in this parser is based
    on how many of the agenda actions are covered.

    This is still WORK IN PROGRESS. We still need to incorporate other linds of losses, including atleast a
    denotation based loss.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 sentence_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 decoder_trainer: DecoderTrainer,
                 decoder_beam_search: BeamSearch,
                 max_decoding_steps: int,
                 action_namespace: str,
                 action_embedding_dim: int,
                 attention_function: SimilarityFunction) -> None:
        super(NlvrSemanticParser, self).__init__(vocab)
        self._sentence_embedder = sentence_embedder
        self._encoder = encoder
        self._decoder_trainer = decoder_trainer
        self._beam_search = decoder_beam_search
        self._max_decoding_steps = max_decoding_steps
        self._action_namespace = action_namespace

        type_productions = types.get_valid_actions(nlvr_types.COMMON_NAME_MAPPING,
                                                   nlvr_types.COMMON_TYPE_SIGNATURE,
                                                   nlvr_types.BASIC_TYPES)
        self._start_index = self.vocab.add_token_to_namespace(START_SYMBOL, self._action_namespace)
        self._end_index = self.vocab.add_token_to_namespace(END_SYMBOL, self._action_namespace)
        self._global_type_productions = defaultdict(list)
        for head, actions in type_productions.items():
            for action in actions:
                action_index = self.vocab.add_token_to_namespace(action, self._action_namespace)
                self._global_type_productions[head].append(action_index)
        self._global_type_productions[START_SYMBOL].append(
                self.vocab.add_token_to_namespace("t", self._action_namespace))
        self._decoder_step = NlvrDecoderStep(vocab=vocab,
                                             action_namespace=action_namespace,
                                             encoder_output_dim=self._encoder.get_output_dim(),
                                             action_embedding_dim=action_embedding_dim,
                                             attention_function=attention_function,
                                             start_index=self._start_index)

    @overrides
    def forward(self,
                sentence: Dict[str, torch.LongTensor],
                agenda: torch.LongTensor,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        """
        Decoder logic for producing type constrained target sequence, that maximize coverage of an agenda.
        """
        # TODO(pradeep): Use labels.
        embedded_input = self._sentence_embedder(sentence)
        sentence_mask = nn_util.get_text_field_mask(sentence).float()
        batch_size = embedded_input.size(0)

        encoder_outputs = self._encoder(embedded_input, sentence_mask)

        final_encoder_output = encoder_outputs[:, -1]  # (batch_size, encoder_output_dim)
        memory_cell = nn_util.new_variable_with_shape(encoder_outputs,
                                                      (batch_size, self._encoder.get_output_dim()), 0)
        attended_sentence = self._decoder_step.attend_on_question(final_encoder_output, encoder_outputs,
                                                                  sentence_mask)
        # (batch_size, agenda_size)
        initial_checklist = nn_util.new_variable_with_shape(agenda, agenda.size(), 0).type(torch.FloatTensor)
        agenda_list = [agenda[i] for i in range(batch_size)]
        initial_checklist_list = [initial_checklist[i] for i in range(batch_size)]
        initial_score_list = [NlvrDecoderStep.score_instance_checklist(checklist) for checklist in
                              initial_checklist_list]
        initial_hidden_state = [final_encoder_output[i] for i in range(batch_size)]
        initial_memory_cell = [memory_cell[i] for i in range(batch_size)]
        initial_attended_sentence = [attended_sentence[i] for i in range(batch_size)]
        encoder_outputs_list = [encoder_outputs[i] for i in range(batch_size)]
        sentence_mask_list = [sentence_mask[i] for i in range(batch_size)]
        initial_state = NlvrDecoderState(agenda_list,
                                         initial_checklist_list,
                                         list(range(batch_size)),
                                         [[] for _ in range(batch_size)],
                                         initial_score_list,
                                         [[START_SYMBOL] for _ in range(batch_size)],
                                         initial_hidden_state,
                                         initial_memory_cell,
                                         initial_attended_sentence,
                                         encoder_outputs_list,
                                         sentence_mask_list,
                                         self._global_type_productions,
                                         self._end_index)

        if self.training:
            # Passing values for target_action_sequences and mask to be able to use MaxMarginalLikelihood
            # trainer. Doesn't matter since we do not use allowed_actions in NlvrDecodeStep anyway. But this
            # is a hack!
            # TODO(pradeep): Change this.
            return self._decoder_trainer.decode(initial_state,
                                                self._decoder_step,
                                                nn_util.new_variable_with_shape(initial_hidden_state[0],
                                                                                torch.Size([1, 1, 1]), 0),
                                                nn_util.new_variable_with_shape(initial_hidden_state[0],
                                                                                torch.Size([1, 1, 1]), 0))
        else:
            raise NotImplementedError

    @classmethod
    def from_params(cls, vocab, params: Params) -> 'NlvrSemanticParser':
        sentence_embedder_params = params.pop("sentence_embedder")
        sentence_embedder = TextFieldEmbedder.from_params(vocab, sentence_embedder_params)
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        max_decoding_steps = params.pop("max_decoding_steps")
        action_namespace = params.pop("action_namespace")
        action_embedding_dim = params.pop("action_embedding_dim", None)
        decoder_trainer = DecoderTrainer.from_params(params.pop("decoder_trainer"))
        decoder_beam_search = BeamSearch.from_params(params.pop("decoder_beam_search"))
        attention_function_type = params.pop("attention_function", None)
        if attention_function_type is not None:
            attention_function = SimilarityFunction.from_params(attention_function_type)
        else:
            attention_function = None
        return cls(vocab,
                   sentence_embedder=sentence_embedder,
                   encoder=encoder,
                   decoder_trainer=decoder_trainer,
                   decoder_beam_search=decoder_beam_search,
                   max_decoding_steps=max_decoding_steps,
                   action_namespace=action_namespace,
                   action_embedding_dim=action_embedding_dim,
                   attention_function=attention_function)


class NlvrDecoderState(WikiTablesDecoderState):
    """
    The two things this state keeps track of, beyond what the ``WikiTablesDecoderState`` does is an
    agenda, which for each instance is a tensor containing the indices of the actions we want to
    appear in the decoded output, and a (soft) checklist indicating how many times each action in an
    agenda has been chosen previously.
    """
    def __init__(self,
                 agenda: List[torch.LongTensor],
                 checklist: List[Variable],
                 *args):
        super(NlvrDecoderState, self).__init__(*args)
        self.agenda = agenda
        self.checklist = checklist

    @classmethod
    def combine_states(cls, states: List['NlvrDecoderState']) -> 'NlvrDecoderState':
        super_state = super(NlvrDecoderState, cls).combine_states(states)
        agenda = [alist for state in states for alist in state.agenda]
        checklist = [clist for state in states for clist in state.checklist]
        return NlvrDecoderState(agenda,
                                checklist,
                                super_state.batch_indices,
                                super_state.action_history,
                                super_state.score,
                                super_state.non_terminal_stack,
                                super_state.hidden_state,
                                super_state.memory_cell,
                                super_state.attended_question,
                                states[0].encoder_outputs,
                                states[0].encoder_output_mask,
                                states[0].global_type_productions,
                                states[0].end_index)

    @overrides
    def _make_new_state_with_group_indices(self, group_indices: List[int]) -> 'NlvrDecoderState':
        super_state = super(NlvrDecoderState, self)._make_new_state_with_group_indices(group_indices)
        group_agenda = [self.agenda[i] for i in group_indices]
        group_checklist = [self.checklist[i] for i in group_indices]
        return NlvrDecoderState(group_agenda,
                                group_checklist,
                                super_state.batch_indices,
                                super_state.action_history,
                                super_state.score,
                                super_state.non_terminal_stack,
                                super_state.hidden_state,
                                super_state.memory_cell,
                                super_state.attended_question,
                                self.encoder_outputs,
                                self.encoder_output_mask,
                                self.global_type_productions,
                                self.end_index)


class NlvrDecoderStep(WikiTablesDecoderStep):
    """
    We just override the ``_compute_new_states`` method here to compute new states based on coverage loss.
    For each action in considered actions, if it is in the agenda, we update the checklist and recalculate the
    score. We'll have to iterate through considered actions in either case because we have to update action
    histories anyway.
    """
    @overrides
    def _compute_new_states(self,
                            state: NlvrDecoderState,
                            log_probs: Variable,
                            hidden_state: Variable,
                            memory_cell: Variable,
                            attended_question: Variable,
                            considered_actions: List[List[int]],
                            allowed_actions: List[Set[int]],
                            max_actions: int = None) -> List[NlvrDecoderState]:
        # TODO(pradeep): We do not have a notion of ``allowed_actions`` for NLVR for now, but this may be
        # applicable in the future.
        probs = torch.exp(log_probs)
        # batch_index -> [(group_index, action, checklist, score)]
        next_states_info: Dict[int, List[Tuple[int, int, Variable, Variable]]] = defaultdict(list)
        for group_index, (batch_index, instance_action_probs) in enumerate(zip(state.batch_indices, probs)):
            instance_agenda = state.agenda[group_index]
            instance_checklist = state.checklist[group_index]
            # action_prob is a Variable.
            for action_index, action_prob in enumerate(instance_action_probs):
                if action_index >= len(considered_actions[group_index]):
                    # Ignoring padding.
                    continue
                # This is the actual index of the action from the original list of actions.
                action = considered_actions[group_index][action_index]
                # If action is not in instance_agenda, mask_variable, and checklist_addition will be all 0s.
                mask_variable = instance_agenda.eq(action)
                checklist_addition = mask_variable.type(torch.FloatTensor) * action_prob
                new_checklist = instance_checklist + checklist_addition
                new_score = self.score_instance_checklist(new_checklist)
                next_states_info[batch_index].append((group_index, action, new_checklist, new_score))
        new_states = []
        for batch_index, states_info in next_states_info.items():
            sorted_states_info = sorted(states_info, key=lambda x: x[3].data.cpu().numpy()[0], reverse=True)
            if max_actions is not None:
                sorted_states_info = sorted_states_info[:max_actions]
            for group_index, action, new_checklist, new_score in sorted_states_info:
                new_action_history = state.action_history[group_index] + [action]
                action_string = self._vocab.get_token_from_index(action, self._action_namespace)
                new_non_terminal_stack = state.update_non_terminal_stack(state.non_terminal_stack[group_index],
                                                                         action_string)
                new_state = NlvrDecoderState([state.agenda[group_index]],
                                             [new_checklist],
                                             [state.batch_indices[group_index]],
                                             [new_action_history],
                                             [new_score],
                                             [new_non_terminal_stack],
                                             [hidden_state[group_index]],
                                             [memory_cell[group_index]],
                                             [attended_question[group_index]],
                                             state.encoder_outputs,
                                             state.encoder_output_mask,
                                             state.global_type_productions,
                                             state.end_index)
                new_states.append(new_state)
        return new_states

    @staticmethod
    def score_instance_checklist(checklist: Variable) -> Variable:
        # TODO(pradeep): Is there something else that's better than mean of agenda probabilities?
        return torch.mean(checklist)
