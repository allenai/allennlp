from typing import List, Dict, Tuple

import torch
from torch.autograd import Variable

from allennlp.data.fields.production_rule_field import ProductionRuleArray
from allennlp.nn.decoding import DecoderState, GrammarState, RnnState
from allennlp.semparse.worlds import NlvrWorld


class NlvrDecoderState(DecoderState['NlvrDecoderState']):
    """
    This class is very similar to ``WikiTablesDecoderState``, except that we keep track of a
    checklist score, and other variables related to it.

    Parameters
    ----------
    batch_indices : ``List[int]``
        Passed to super class; see docs there.
    action_history : ``List[List[int]]``
        Passed to super class; see docs there.
    score : ``List[torch.Tensor]``
        Passed to super class; see docs there.
    rnn_state : ``List[RnnState]``
        An ``RnnState`` for every group element.  This keeps track of the current decoder hidden
        state, the previous decoder output, the output from the encoder (for computing attentions),
        and other things that are typical seq2seq decoder state things.
    grammar_state : ``List[GrammarState]``
        This hold the current grammar state for each element of the group.  The ``GrammarState``
        keeps track of which actions are currently valid.
    action_embeddings : ``torch.Tensor``
        The global action embeddings tensor.  Has shape ``(num_global_embeddable_actions,
        action_embedding_dim)``.
    action_indices : ``Dict[Tuple[int, int], int]``
        A mapping from ``(batch_index, action_index)`` to ``global_action_index``.
    possible_actions : ``List[List[ProductionRuleArray]]``
        The list of all possible actions that was passed to ``model.forward()``.  We need this so
        we can recover production strings, which we need to update grammar states.
    worlds : ``List[List[NlvrWorld]]``
        The worlds associated with each element. These are needed to compute the denotations. The
        outer list corresponds to elements, and the inner corresponds to worlds related to each
        element.
    label_strings : ``List[List[str]]``
        String representations of labels for the elements provided. When scoring finished states, we
        will compare the denotations of their action sequences against these labels. For each
        element, there will be as many labels as there are worlds.
    terminal_actions : ``List[torch.Tensor]``, optional
        Each element in the list is a vector containing the indices of terminal actions. Currently
        the vectors are the same for all instances, because we consider all terminals for each
        instance. In the future, we may want to include only world-specific terminal actions here.
        Each of these vectors is needed for computing checklists for next states, only if this state
        is being while training a parser without logical forms.
    checklist_target : ``List[torch.LongTensor]``, optional
        List of targets corresponding to agendas that indicate the states we want the checklists to
        ideally be. Each element in this list is the same size as the corresponding element in
        ``agenda_relevant_actions``, and it contains 1 for each corresponding action in the relevant
        actions list that we want to see in the final logical form, and 0 for each corresponding
        action that we do not. Needed only if this state is being used while training a parser
        without logical forms.
    checklist_masks : ``List[torch.Tensor]``, optional
        Masks corresponding to ``terminal_actions``, indicating which of those actions are relevant
        for checklist computation. For example, if the parser is penalizing non-agenda terminal
        actions, all the terminal actions are relevant. Needed only if this state is being used
        while training a parser without logical forms.
    checklist : ``List[Variable]``, optional
        A checklist for each instance indicating how many times each action in its agenda has
        been chosen previously. It contains the actual counts of the agenda actions. Needed only if
        this state is being used while training a parser without logical forms.
    """
    # TODO(pradeep): Group checklist related pieces into a checklist state.
    def __init__(self,
                 batch_indices: List[int],
                 action_history: List[List[int]],
                 score: List[torch.Tensor],
                 rnn_state: List[RnnState],
                 grammar_state: List[GrammarState],
                 action_embeddings: torch.Tensor,
                 action_indices: Dict[Tuple[int, int], int],
                 possible_actions: List[List[ProductionRuleArray]],
                 worlds: List[List[NlvrWorld]],
                 label_strings: List[List[str]],
                 terminal_actions: List[torch.Tensor] = None,
                 checklist_target: List[torch.Tensor] = None,
                 checklist_masks: List[torch.Tensor] = None,
                 checklist: List[Variable] = None) -> None:
        super(NlvrDecoderState, self).__init__(batch_indices, action_history, score)
        self.rnn_state = rnn_state
        self.grammar_state = grammar_state
        self.terminal_actions = terminal_actions
        self.checklist_target = checklist_target
        self.checklist_mask = checklist_masks
        self.checklist = checklist
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
        batch_indices = [batch_index for state in states for batch_index in state.batch_indices]
        action_histories = [action_history for state in states for action_history in state.action_history]
        scores = [score for state in states for score in state.score]
        rnn_states = [rnn_state for state in states for rnn_state in state.rnn_state]
        grammar_states = [grammar_state for state in states for grammar_state in state.grammar_state]
        if states[0].terminal_actions is not None:
            terminal_actions = [actions for state in states for actions in state.terminal_actions]
            checklist_target = [target_list for state in states for target_list in
                                state.checklist_target]
            checklist_masks = [mask for state in states for mask in state.checklist_mask]
            checklist = [checklist_list for state in states for checklist_list in state.checklist]
        else:
            terminal_actions = None
            checklist_target = None
            checklist_masks = None
            checklist = None
        return NlvrDecoderState(batch_indices=batch_indices,
                                action_history=action_histories,
                                score=scores,
                                rnn_state=rnn_states,
                                grammar_state=grammar_states,
                                action_embeddings=states[0].action_embeddings,
                                action_indices=states[0].action_indices,
                                possible_actions=states[0].possible_actions,
                                worlds=states[0].worlds,
                                label_strings=states[0].label_strings,
                                terminal_actions=terminal_actions,
                                checklist_target=checklist_target,
                                checklist_masks=checklist_masks,
                                checklist=checklist)
