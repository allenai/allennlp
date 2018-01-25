from typing import List, Set, Dict, Tuple
from collections import defaultdict

from overrides import overrides

import torch
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.data.fields.production_rule_field import ProductionRuleArray
from allennlp.data.semparse.type_declarations import GrammarState
from allennlp.data.semparse.worlds import NlvrWorld
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.nn.decoding import DecoderTrainer, BeamSearch
from allennlp.nn import util as nn_util
from allennlp.models.model import Model
from allennlp.models.encoder_decoders.wikitables_semantic_parser import WikiTablesDecoderStep
from allennlp.models.encoder_decoders.wikitables_semantic_parser import WikiTablesDecoderState
from allennlp.models.encoder_decoders.wikitables_semantic_parser import WikiTablesSemanticParser


@Model.register("nlvr_parser")
class NlvrSemanticParser(WikiTablesSemanticParser):
    """
    ``NlvrSemanticParser`` is a semantic parsing model built for the NLVR domain.  There is a lot of
    overlap with ``WikiTablesSemanticParser`` here. We may want to eventually move the common
    functionality into a more general transition-based parsing class.

    The main differences between this parser and what we have for Wikitables are that we have an
    agenda of actions instead of complete target action sequences, and accordingly the score in this
    parser is based on how many of the agenda actions are covered.

    This is still WORK IN PROGRESS. We still need to incorporate other kinds of losses, including
    atleast a denotation based loss.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 sentence_embedder: TextFieldEmbedder,
                 nonterminal_embedder: TextFieldEmbedder,
                 terminal_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 decoder_trainer: DecoderTrainer,
                 decoder_beam_search: BeamSearch,
                 max_decoding_steps: int,
                 attention_function: SimilarityFunction) -> None:
        # Forcing embed_terminals to be True because we are not doing entity linking for NLVR
        # (atleast for now).
        super(NlvrSemanticParser, self).__init__(vocab=vocab,
                                                 question_embedder=sentence_embedder,
                                                 nonterminal_embedder=nonterminal_embedder,
                                                 terminal_embedder=terminal_embedder,
                                                 encoder=encoder,
                                                 decoder_trainer=decoder_trainer,
                                                 decoder_beam_search=decoder_beam_search,
                                                 max_decoding_steps=max_decoding_steps,
                                                 attention_function=attention_function,
                                                 embed_terminals=True)

        action_embedding_dim = nonterminal_embedder.get_output_dim() * 2

        # self._decoder_step would be set to ``WikiTablesDecoderStep``. Rewriting it.
        self._decoder_step = NlvrDecoderStep(encoder_output_dim=self._encoder.get_output_dim(),
                                             action_embedding_dim=action_embedding_dim,
                                             attention_function=attention_function)
    @overrides
    def forward(self,
                sentence: Dict[str, torch.LongTensor],
                world: List[NlvrWorld],
                actions: List[List[ProductionRuleArray]],
                agenda: torch.LongTensor,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ,unused-argument
        """
        Decoder logic for producing type constrained target sequences, that maximize coverage of
        their respective agendas. This will change soon, to include a denotation based score as
        well, once we have a way to transform action sequences into logical forms that can be
        executed to produce denotations.
        """
        # TODO(pradeep): Use labels.
        embedded_input = self._question_embedder(sentence)
        sentence_mask = nn_util.get_text_field_mask(sentence).float()

        batch_size = embedded_input.size(0)

        encoder_outputs = self._encoder(embedded_input, sentence_mask)

        final_encoder_output = encoder_outputs[:, -1]  # (batch_size, encoder_output_dim)
        memory_cell = nn_util.new_variable_with_size(encoder_outputs,
                                                     (batch_size, self._encoder.get_output_dim()),
                                                     0)
        attended_sentence, _ = self._decoder_step.attend_on_question(final_encoder_output,
                                                                     encoder_outputs, sentence_mask)
        action_embeddings, action_indices, initial_action_embedding = self._embed_actions(actions)
        # Get a mapping from production rules to global action ids.
        agenda_mask = agenda != -1
        # (batch_size, agenda_size)
        initial_checklist = nn_util.new_variable_with_size(agenda, agenda.size(), 0).float()
        agenda_list = [agenda[i] for i in range(batch_size)]
        agenda_mask_list = [agenda_mask[i] for i in range(batch_size)]
        initial_checklist_list = [initial_checklist[i] for i in range(batch_size)]
        initial_score_list = [NlvrDecoderStep.score_instance_checklist(checklist, agenda_mask)
                              for checklist, agenda_mask in zip(initial_checklist_list,
                                                                agenda_mask_list)]
        initial_hidden_state = [final_encoder_output[i] for i in range(batch_size)]
        initial_memory_cell = [memory_cell[i] for i in range(batch_size)]
        initial_action_embedding_list = [initial_action_embedding for _ in range(batch_size)]
        initial_grammar_state = [self._create_grammar_state(world[i], actions[i]) for i in
                                 range(batch_size)]
        initial_attended_sentence = [attended_sentence[i] for i in range(batch_size)]
        encoder_outputs_list = [encoder_outputs[i] for i in range(batch_size)]
        sentence_mask_list = [sentence_mask[i] for i in range(batch_size)]
        initial_state = NlvrDecoderState(agenda_list,
                                         agenda_mask_list,
                                         initial_checklist_list,
                                         list(range(batch_size)),
                                         [[] for _ in range(batch_size)],
                                         initial_score_list,
                                         initial_hidden_state,
                                         initial_memory_cell,
                                         initial_action_embedding_list,
                                         initial_attended_sentence,
                                         initial_grammar_state,
                                         encoder_outputs_list,
                                         sentence_mask_list,
                                         action_embeddings,
                                         action_indices,
                                         actions)

        outputs = self._decoder_trainer.decode(initial_state, self._decoder_step,  # type: ignore
                                               self._max_decoding_steps)
        if not self.training:
            best_final_states = self._beam_search.search(self._max_decoding_steps, initial_state,
                                                         self._decoder_step)
            best_action_sequences = [best_final_states[i][0].action_history for i in
                                     range(batch_size)]
            outputs['best_action_sequence'] = best_action_sequences
        return outputs

    @classmethod
    def from_params(cls, vocab, params: Params) -> 'NlvrSemanticParser':
        sentence_embedder_params = params.pop("sentence_embedder")
        sentence_embedder = TextFieldEmbedder.from_params(vocab, sentence_embedder_params)
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        max_decoding_steps = params.pop("max_decoding_steps")
        nonterminal_embedder = TextFieldEmbedder.from_params(vocab, params.pop("nonterminal_embedder"))
        terminal_embedder = TextFieldEmbedder.from_params(vocab, params.pop("terminal_embedder"))
        decoder_trainer = DecoderTrainer.from_params(params.pop("decoder_trainer"))
        decoder_beam_search = BeamSearch.from_params(params.pop("decoder_beam_search"))
        attention_function_type = params.pop("attention_function", None)
        if attention_function_type is not None:
            attention_function = SimilarityFunction.from_params(attention_function_type)
        else:
            attention_function = None
        params.assert_empty(cls.__name__)
        return cls(vocab,
                   sentence_embedder=sentence_embedder,
                   nonterminal_embedder=nonterminal_embedder,
                   terminal_embedder=terminal_embedder,
                   encoder=encoder,
                   decoder_trainer=decoder_trainer,
                   decoder_beam_search=decoder_beam_search,
                   max_decoding_steps=max_decoding_steps,
                   attention_function=attention_function)


class NlvrDecoderState(WikiTablesDecoderState):
    """
    The three things this state keeps track of, beyond what the ``WikiTablesDecoderState`` does is
    1) an agenda, which for each instance is a tensor containing the indices of the actions we want
    to see in the decoded output
    2) a mask corresponding to the agenda, that shows which elements are not padding; and
    3) a (soft) checklist indicating how many times each action in an agenda has been chosen
    previously

    The checklist is soft because it contains the (sum of) the probabilities previously assigned to
    each action.
    """
    def __init__(self,
                 agenda: List[torch.LongTensor],
                 agenda_mask: List[torch.LongTensor],
                 checklist: List[Variable],
                 batch_indices: List[int],
                 action_history: List[List[int]],
                 score: List[torch.Tensor],
                 hidden_state: List[torch.Tensor],
                 memory_cell: List[torch.Tensor],
                 previous_action_embedding: List[torch.Tensor],
                 attended_sentence: List[torch.Tensor],
                 grammar_state: List[GrammarState],
                 encoder_outputs: List[torch.Tensor],
                 encoder_output_mask: List[torch.Tensor],
                 action_embeddings: torch.Tensor,
                 action_indices: Dict[Tuple[int, int], int],
                 possible_actions: List[List[ProductionRuleArray]]) -> None:
        super(NlvrDecoderState, self).__init__(batch_indices=batch_indices,
                                               action_history=action_history,
                                               score=score,
                                               hidden_state=hidden_state,
                                               memory_cell=memory_cell,
                                               previous_action_embedding=previous_action_embedding,
                                               attended_question=attended_sentence,
                                               grammar_state=grammar_state,
                                               encoder_outputs=encoder_outputs,
                                               encoder_output_mask=encoder_output_mask,
                                               action_embeddings=action_embeddings,
                                               action_indices=action_indices,
                                               possible_actions=possible_actions,
                                               flattened_linking_scores=None,
                                               actions_to_entities=None,
                                               entity_types=None)
        self.agenda = agenda
        self.agenda_mask = agenda_mask
        self.checklist = checklist

    @classmethod
    def combine_states(cls, states) -> 'NlvrDecoderState':
        super_state = super(NlvrDecoderState, cls).combine_states(states)
        agenda = [agenda_list for state in states for agenda_list in state.agenda]
        agenda_mask = [mask_list for state in states for mask_list in state.agenda_mask]
        checklist = [checklist_list for state in states for checklist_list in state.checklist]
        return NlvrDecoderState(agenda,
                                agenda_mask,
                                checklist,
                                super_state.batch_indices,
                                super_state.action_history,
                                super_state.score,
                                super_state.hidden_state,
                                super_state.memory_cell,
                                super_state.previous_action_embedding,
                                super_state.attended_question,
                                super_state.grammar_state,
                                states[0].encoder_outputs,
                                states[0].encoder_output_mask,
                                states[0].action_embeddings,
                                states[0].action_indices,
                                states[0].possible_actions)

    @overrides
    def _make_new_state_with_group_indices(self, group_indices: List[int]) -> 'NlvrDecoderState':
        super_state = super(NlvrDecoderState, self)._make_new_state_with_group_indices(group_indices)
        group_agenda = [self.agenda[i] for i in group_indices]
        group_agenda_mask = [self.agenda_mask[i] for i in group_indices]
        group_checklist = [self.checklist[i] for i in group_indices]
        return NlvrDecoderState(group_agenda,
                                group_agenda_mask,
                                group_checklist,
                                super_state.batch_indices,
                                super_state.action_history,
                                super_state.score,
                                super_state.hidden_state,
                                super_state.memory_cell,
                                super_state.previous_action_embedding,
                                super_state.attended_question,
                                super_state.grammar_state,
                                self.encoder_outputs,
                                self.encoder_output_mask,
                                self.action_embeddings,
                                self.action_indices,
                                self.possible_actions)


class NlvrDecoderStep(WikiTablesDecoderStep):
    def __init__(self,
                 encoder_output_dim: int,
                 action_embedding_dim: int,
                 attention_function: SimilarityFunction) -> None:
        # Setting num_entity_types to 0 since that is needed only for linking, and we
        # will not use linking in NLVR (yet).
        super(NlvrDecoderStep, self).__init__(encoder_output_dim=encoder_output_dim,
                                              action_embedding_dim=action_embedding_dim,
                                              attention_function=attention_function,
                                              num_entity_types=0)

    @classmethod
    def _compute_new_states(cls,  # type: ignore
                            state: NlvrDecoderState,
                            log_probs: torch.Tensor,
                            hidden_state: torch.Tensor,
                            memory_cell: torch.Tensor,
                            action_embeddings: torch.Tensor,
                            attended_question: torch.Tensor,
                            considered_actions: List[List[int]],
                            allowed_actions: List[Set[int]],
                            max_actions: int = None) -> List[NlvrDecoderState]:
        """
        We override the ``_compute_new_states`` method here to compute new states based on coverage
        loss.  For each action in considered actions, if it is in the agenda, we update the checklist
        and recalculate the score. We'll have to iterate through considered actions in either case
        because we have to update action histories anyway.
        """
        # TODO(pradeep): We do not have a notion of ``allowed_actions`` for NLVR for now, but this
        # may be applicable in the future.
        # Note that we're not sorting these probs. We'll score checklists and sort states based on
        # checklist scores later.
        probs = torch.exp(log_probs)
        # batch_index -> [(group_index, action_index, action, checklist, score)]
        next_states_info: Dict[int, List[Tuple[int, int, int, Variable, Variable]]] = defaultdict(list)
        for group_index, (batch_index, instance_action_probs) in enumerate(zip(state.batch_indices, probs)):
            instance_agenda = state.agenda[group_index]  # (agenda_size,)
            instance_agenda_mask = state.agenda_mask[group_index]  # (agenda_size,)
            instance_checklist = state.checklist[group_index]  # (agenda_size,)
            # action_prob is a Variable.
            for action_index, action_prob in enumerate(instance_action_probs):
                # This is the actual index of the action from the original list of actions.
                action = considered_actions[group_index][action_index]
                if action == -1:
                    # Ignoring padding.
                    continue
                # If action is not in instance_agenda, mask_variable, and checklist_addition will be
                # all 0s.
                checklist_mask = instance_agenda == action  # (agenda_size,)
                checklist_addition = checklist_mask.float() * action_prob  # (agenda_size,)
                new_checklist = instance_checklist + checklist_addition  # (agenda_size,)
                new_score = cls.score_instance_checklist(new_checklist, instance_agenda_mask)
                next_states_info[batch_index].append((group_index, action_index, action,
                                                      new_checklist, new_score))
        new_states = []
        for batch_index, states_info in next_states_info.items():
            # We need to sort states by scores. We batch all the scores first for efficient sorting.
            batch_scores = torch.cat([state_info[-1] for state_info in states_info])
            _, sorted_indices = batch_scores.sort(-1, descending=True)
            sorted_states_info = [states_info[i] for i in sorted_indices.cpu().data.numpy()]
            if max_actions is not None:
                sorted_states_info = sorted_states_info[:max_actions]
            for group_index, action_index, action, new_checklist, new_score in sorted_states_info:
                new_action_history = state.action_history[group_index] + [action]
                action_embedding = action_embeddings[group_index, action_index, :]
                left_side = state.possible_actions[batch_index][action]['left'][0]
                right_side = state.possible_actions[batch_index][action]['right'][0]
                new_grammar_state = state.grammar_state[group_index].take_action(left_side,
                                                                                 right_side)

                new_state = NlvrDecoderState([state.agenda[group_index]],
                                             [state.agenda_mask[group_index]],
                                             [new_checklist],
                                             [batch_index],
                                             [new_action_history],
                                             [new_score],
                                             [hidden_state[group_index]],
                                             [memory_cell[group_index]],
                                             [action_embedding],
                                             [attended_question[group_index]],
                                             [new_grammar_state],
                                             state.encoder_outputs,
                                             state.encoder_output_mask,
                                             state.action_embeddings,
                                             state.action_indices,
                                             state.possible_actions)
                new_states.append(new_state)
        return new_states

    @staticmethod
    def score_instance_checklist(checklist: Variable, agenda_mask: Variable) -> Variable:
        """
        Takes a checklist and agenda's mask (that shows which of the agenda items are not actually
        padding), and scores the checklist. For now, the score is simply the average of checklist
        probabilities.
        """
        # TODO(pradeep): Is there a better alternative to mean of agenda probabilities?
        float_mask = agenda_mask.float()
        return torch.sum(checklist * float_mask) / float_mask.sum()
