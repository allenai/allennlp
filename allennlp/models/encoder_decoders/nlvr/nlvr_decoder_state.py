from typing import List, Dict, Tuple

import torch
from torch.autograd import Variable

from allennlp.data.fields.production_rule_field import ProductionRuleArray
from allennlp.nn.decoding import DecoderState
from allennlp.semparse.type_declarations import GrammarState
from allennlp.semparse.worlds import NlvrWorld


class NlvrDecoderState(DecoderState['NlvrDecoderState']):
    """
    This class is very similar to ``WikiTablesDecoderState``, except that we keep track of a
    checklist score, and other variables related to it.

    Parameters
    ----------
    terminal_actions : ``List[torch.Tensor]``
        Each element in the list is a vector containing the indices of terminal actions. Currently
        the vectors are the same for all instances, because we consider all terminals for each
        instance. In the future, we may want to include only world-specific terminal actions here.
        Each of these vectors is needed for computing checklists for next states.
    checklist_target : ``List[torch.LongTensor]``
        List of targets corresponding to agendas that indicate the states we want the checklists to
        ideally be. Each element in this list is the same size as the corresponding element in
        ``agenda_relevant_actions``, and it contains 1 for each corresponding action in the relevant
        actions list that we want to see in the final logical form, and 0 for each corresponding
        action that we do not.
    checklist_masks : ``List[torch.Tensor]``
        Masks corresponding to ``terminal_actions``, indicating which of those actions are relevant
        for checklist computation. For example, if the parser is penalizing non-agenda terminal
        actions, all the terminal actions are relevant.
    checklist : ``List[Variable]``
        A checklist for each instance indicating how many times each action in its agenda has
        been chosen previously. It contains the actual counts of the agenda actions.
    batch_indices : ``List[int]``
        Passed to super class; see docs there.
    action_history : ``List[List[int]]``
        Passed to super class; see docs there.
    score : ``List[torch.Tensor]``
        Passed to super class; see docs there.
    hidden_state : ``List[torch.Tensor]``
        This holds the LSTM hidden state for each element of the group.  Each tensor has shape
        ``(decoder_output_dim,)``.
    memory_cell : ``List[torch.Tensor]``
        This holds the LSTM memory cell for each element of the group.  Each tensor has shape
        ``(decoder_output_dim,)``.
    previous_action_embedding : ``List[torch.Tensor]``
        This holds the embedding for the action we took at the last timestep (which gets input to
        the decoder).  Each tensor has shape ``(action_embedding_dim,)``.
    attended_sentence : ``List[torch.Tensor]``
        This holds the attention-weighted sum over the sentence representations that we computed in
        the previous timestep, for each element in the group.  We keep this as part of the state
        because we use the previous attention as part of our decoder cell update.  Each tensor in
        this list has shape ``(encoder_output_dim,)``.
    grammar_state : ``List[GrammarState]``
        This hold the current grammar state for each element of the group.  The ``GrammarState``
        keeps track of which actions are currently valid.
    encoder_outputs : ``List[torch.Tensor]``
        A list of variables, each of shape ``(sentence_length, encoder_output_dim)``, containing
        the encoder outputs at each timestep.  The list is over batch elements, and we do the input
        this way so we can easily do a ``torch.cat`` on a list of indices into this batched list.

        Note that all of the above lists are of length ``group_size``, while the encoder outputs
        and mask are lists of length ``batch_size``.  We always pass around the encoder outputs and
        mask unmodified, regardless of what's in the grouping for this state.  We'll use the
        ``batch_indices`` for the group to pull pieces out of these lists when we're ready to
        actually do some computation.
    encoder_output_mask : ``List[torch.Tensor]``
        A list of variables, each of shape ``(sentence_length,)``, containing a mask over sentence
        tokens for each batch instance.  This is a list over batch elements, for the same reasons
        as above.
    action_embeddings : ``torch.Tensor``
        The global action embeddings tensor.  Has shape ``(num_global_embeddable_actions,
        action_embedding_dim)``.
    action_indices : ``Dict[Tuple[int, int], int]``
        A mapping from ``(batch_index, action_index)`` to ``global_action_index``.
    possible_actions : ``List[List[ProductionRuleArray]]``
        The list of all possible actions that was passed to ``model.forward()``.  We need this so
        we can recover production strings, which we need to update grammar states.
    world : ``List[NlvrWorld]``
        The world associated with each element. This is needed to compute the denotations.
    label_strings : ``List[str]``
        String representations of labels for the elements provided. When scoring finished states, we
        will compare the denotations of their action sequences against these labels.
    """
    def __init__(self,
                 terminal_actions: List[torch.Tensor],
                 checklist_target: List[torch.Tensor],
                 checklist_masks: List[torch.Tensor],
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
                 possible_actions: List[List[ProductionRuleArray]],
                 worlds: List[NlvrWorld],
                 label_strings: List[str]) -> None:
        super(NlvrDecoderState, self).__init__(batch_indices, action_history, score)
        self.terminal_actions = terminal_actions
        self.checklist_target = checklist_target
        self.checklist_mask = checklist_masks
        self.checklist = checklist
        self.hidden_state = hidden_state
        self.memory_cell = memory_cell
        self.previous_action_embedding = previous_action_embedding
        self.attended_sentence = attended_sentence
        self.grammar_state = grammar_state
        self.encoder_outputs = encoder_outputs
        self.encoder_output_mask = encoder_output_mask
        self.action_embeddings = action_embeddings
        self.action_indices = action_indices
        self.possible_actions = possible_actions
        self.worlds = worlds
        self.label_strings = label_strings

    def get_valid_actions(self) -> List[List[int]]:
        """
        Returns a list of valid actions for each element of the group.
        """
        valid_actions = [state.get_valid_actions() for state in self.grammar_state]
        return valid_actions

    def is_finished(self) -> bool:
        """This method is identical to ``WikiTablesDecoderState.is_finished``."""
        if len(self.batch_indices) != 1:
            raise RuntimeError("is_finished() is only defined with a group_size of 1")
        return self.grammar_state[0].is_finished()

    @classmethod
    def combine_states(cls, states) -> 'NlvrDecoderState':
        terminal_actions = [actions for state in states for actions in state.terminal_actions]
        checklist_target = [target_list for state in states for target_list in
                            state.checklist_target]
        checklist_masks = [mask for state in states for mask in state.checklist_mask]
        checklist = [checklist_list for state in states for checklist_list in state.checklist]
        batch_indices = [batch_index for state in states for batch_index in state.batch_indices]
        action_histories = [action_history for state in states for action_history in state.action_history]
        scores = [score for state in states for score in state.score]
        hidden_states = [hidden_state for state in states for hidden_state in state.hidden_state]
        memory_cells = [memory_cell for state in states for memory_cell in state.memory_cell]
        previous_action = [action for state in states for action in state.previous_action_embedding]
        attended_sentence = [attended for state in states for attended in state.attended_sentence]
        grammar_states = [grammar_state for state in states for grammar_state in state.grammar_state]
        return NlvrDecoderState(terminal_actions,
                                checklist_target,
                                checklist_masks,
                                checklist,
                                batch_indices,
                                action_histories,
                                scores,
                                hidden_states,
                                memory_cells,
                                previous_action,
                                attended_sentence,
                                grammar_states,
                                states[0].encoder_outputs,
                                states[0].encoder_output_mask,
                                states[0].action_embeddings,
                                states[0].action_indices,
                                states[0].possible_actions,
                                states[0].worlds,
                                states[0].label_strings)
