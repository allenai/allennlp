from typing import Any, List, Sequence

import torch

from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.nn import util
from allennlp.state_machines.states.checklist_statelet import ChecklistStatelet
from allennlp.state_machines.states.grammar_based_state import GrammarBasedState
from allennlp.state_machines.states.grammar_statelet import GrammarStatelet
from allennlp.state_machines.states.rnn_statelet import RnnStatelet


class CoverageState(GrammarBasedState):
    """
    This ``State`` adds one field to a ``GrammarBasedState``: a ``ChecklistStatelet``
    that is used to specify a set of actions that should be taken during decoder, and keep track of
    which of those actions have already been selected.

    We only provide documentation for the ``ChecklistStatelet`` here; for the rest, see
    :class:`GrammarBasedState`.

    Parameters
    ----------
    batch_indices : ``List[int]``
    action_history : ``List[List[int]]``
    score : ``List[torch.Tensor]``
    rnn_state : ``List[RnnStatelet]``
    grammar_state : ``List[GrammarStatelet]``
    checklist_state : ``List[ChecklistStatelet]``
        This holds the current checklist state for each element of the group.  The
        ``ChecklistStatelet`` keeps track of which actions are preferred by some agenda, and which
        of those have already been selected during decoding.
    possible_actions : ``List[List[ProductionRule]]``
    extras : ``List[Any]``, optional (default=None)
    debug_info : ``List[Any]``, optional (default=None).
    """
    def __init__(self,
                 batch_indices: List[int],
                 action_history: List[List[int]],
                 score: List[torch.Tensor],
                 rnn_state: List[RnnStatelet],
                 grammar_state: List[GrammarStatelet],
                 checklist_state: List[ChecklistStatelet],
                 possible_actions: List[List[ProductionRule]],
                 extras: List[Any] = None,
                 debug_info: List = None) -> None:
        super().__init__(batch_indices=batch_indices,
                         action_history=action_history,
                         score=score,
                         rnn_state=rnn_state,
                         grammar_state=grammar_state,
                         possible_actions=possible_actions,
                         extras=extras,
                         debug_info=debug_info)
        self.checklist_state = checklist_state

    def new_state_from_group_index(self,
                                   group_index: int,
                                   action: int,
                                   new_score: torch.Tensor,
                                   new_rnn_state: RnnStatelet,
                                   considered_actions: List[int] = None,
                                   action_probabilities: List[float] = None,
                                   attention_weights: torch.Tensor = None) -> 'CoverageState':
        super_class_state = super().new_state_from_group_index(group_index=group_index,
                                                               action=action,
                                                               new_score=new_score,
                                                               new_rnn_state=new_rnn_state,
                                                               considered_actions=considered_actions,
                                                               action_probabilities=action_probabilities,
                                                               attention_weights=attention_weights)
        new_checklist = self.checklist_state[group_index].update(action)
        return CoverageState(batch_indices=super_class_state.batch_indices,
                             action_history=super_class_state.action_history,
                             score=super_class_state.score,
                             rnn_state=super_class_state.rnn_state,
                             grammar_state=super_class_state.grammar_state,
                             checklist_state=[new_checklist],
                             possible_actions=super_class_state.possible_actions,
                             extras=super_class_state.extras,
                             debug_info=super_class_state.debug_info)

    @classmethod
    def combine_states(cls, states: Sequence['CoverageState']) -> 'CoverageState':  # type: ignore
        super_class_state = super().combine_states(states)
        checklist_states = [checklist_state for state in states for checklist_state in state.checklist_state]
        return CoverageState(batch_indices=super_class_state.batch_indices,
                             action_history=super_class_state.action_history,
                             score=super_class_state.score,
                             rnn_state=super_class_state.rnn_state,
                             grammar_state=super_class_state.grammar_state,
                             checklist_state=checklist_states,
                             possible_actions=super_class_state.possible_actions,
                             extras=super_class_state.extras,
                             debug_info=super_class_state.debug_info)

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return all([
                    self.batch_indices == other.batch_indices,
                    self.action_history == other.action_history,
                    util.tensors_equal(self.score, other.score, tolerance=1e-3),
                    util.tensors_equal(self.rnn_state, other.rnn_state, tolerance=1e-4),
                    self.grammar_state == other.grammar_state,
                    self.checklist_state == other.checklist_state,
                    self.possible_actions == other.possible_actions,
                    self.extras == other.extras,
                    util.tensors_equal(self.debug_info, other.debug_info, tolerance=1e-6),
                    ])
        return NotImplemented
