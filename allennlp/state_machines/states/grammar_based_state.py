from typing import Any, Dict, List, Sequence, Tuple

import torch

from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.state_machines.states.grammar_statelet import GrammarStatelet
from allennlp.state_machines.states.rnn_statelet import RnnStatelet
from allennlp.state_machines.states.state import State

# This syntax is pretty weird and ugly, but it's necessary to make mypy happy with the API that
# we've defined.  We're using generics to make the type of `combine_states` come out right.  See
# the note in `state_machines.state.py` for a little more detail.
class GrammarBasedState(State['GrammarBasedState']):
    """
    A generic State that's suitable for most models that do grammar-based decoding.  We keep around
    a `group` of states, and each element in the group has a few things: a batch index, an action
    history, a score, an ``RnnStatelet``, and a ``GrammarStatelet``.  We additionally have some
    information that's independent of any particular group element: a list of all possible actions
    for all batch instances passed to ``model.forward()``, and a ``extras`` field that you can use
    if you really need some extra information about each batch instance (like a string description,
    or other metadata).

    Finally, we also have a specially-treated, optional ``debug_info`` field.  If this is given, it
    should be an empty list for each group instance when the initial state is created.  In that
    case, we will keep around information about the actions considered at each timestep of decoding
    and other things that you might want to visualize in a demo.  This probably isn't necessary for
    training, and to get it right we need to copy a bunch of data structures for each new state, so
    it's best used only at evaluation / demo time.

    Parameters
    ----------
    batch_indices : ``List[int]``
        Passed to super class; see docs there.
    action_history : ``List[List[int]]``
        Passed to super class; see docs there.
    score : ``List[torch.Tensor]``
        Passed to super class; see docs there.
    rnn_state : ``List[RnnStatelet]``
        An ``RnnStatelet`` for every group element.  This keeps track of the current decoder hidden
        state, the previous decoder output, the output from the encoder (for computing attentions),
        and other things that are typical seq2seq decoder state things.
    grammar_state : ``List[GrammarStatelet]``
        This hold the current grammar state for each element of the group.  The ``GrammarStatelet``
        keeps track of which actions are currently valid.
    possible_actions : ``List[List[ProductionRule]]``
        The list of all possible actions that was passed to ``model.forward()``.  We need this so
        we can recover production strings, which we need to update grammar states.
    extras : ``List[Any]``, optional (default=None)
        If you need to keep around some extra data for each instance in the batch, you can put that
        in here, without adding another field.  This should be used `very sparingly`, as there is
        no type checking or anything done on the contents of this field, and it will just be passed
        around between ``States`` as-is, without copying.
    debug_info : ``List[Any]``, optional (default=None).
    """
    def __init__(self,
                 batch_indices: List[int],
                 action_history: List[List[int]],
                 score: List[torch.Tensor],
                 rnn_state: List[RnnStatelet],
                 grammar_state: List[GrammarStatelet],
                 possible_actions: List[List[ProductionRule]],
                 extras: List[Any] = None,
                 debug_info: List = None) -> None:
        super().__init__(batch_indices, action_history, score)
        self.rnn_state = rnn_state
        self.grammar_state = grammar_state
        self.possible_actions = possible_actions
        self.extras = extras
        self.debug_info = debug_info

    def new_state_from_group_index(self,
                                   group_index: int,
                                   action: int,
                                   new_score: torch.Tensor,
                                   new_rnn_state: RnnStatelet,
                                   considered_actions: List[int] = None,
                                   action_probabilities: List[float] = None,
                                   attention_weights: torch.Tensor = None) -> 'GrammarBasedState':
        batch_index = self.batch_indices[group_index]
        new_action_history = self.action_history[group_index] + [action]
        production_rule = self.possible_actions[batch_index][action][0]
        new_grammar_state = self.grammar_state[group_index].take_action(production_rule)
        if self.debug_info is not None:
            attention = attention_weights[group_index] if attention_weights is not None else None
            debug_info = {
                    'considered_actions': considered_actions,
                    'question_attention': attention,
                    'probabilities': action_probabilities,
                    }
            new_debug_info = [self.debug_info[group_index] + [debug_info]]
        else:
            new_debug_info = None
        return GrammarBasedState(batch_indices=[batch_index],
                                 action_history=[new_action_history],
                                 score=[new_score],
                                 rnn_state=[new_rnn_state],
                                 grammar_state=[new_grammar_state],
                                 possible_actions=self.possible_actions,
                                 extras=self.extras,
                                 debug_info=new_debug_info)

    def print_action_history(self, group_index: int = None) -> None:
        scores = self.score if group_index is None else [self.score[group_index]]
        batch_indices = self.batch_indices if group_index is None else [self.batch_indices[group_index]]
        histories = self.action_history if group_index is None else [self.action_history[group_index]]
        for score, batch_index, action_history in zip(scores, batch_indices, histories):
            print('  ', score.detach().cpu().numpy()[0],
                  [self.possible_actions[batch_index][action][0] for action in action_history])

    def get_valid_actions(self) -> List[Dict[str, Tuple[torch.Tensor, torch.Tensor, List[int]]]]:
        """
        Returns a list of valid actions for each element of the group.
        """
        return [state.get_valid_actions() for state in self.grammar_state]

    def is_finished(self) -> bool:
        if len(self.batch_indices) != 1:
            raise RuntimeError("is_finished() is only defined with a group_size of 1")
        return self.grammar_state[0].is_finished()

    @classmethod
    def combine_states(cls, states: Sequence['GrammarBasedState']) -> 'GrammarBasedState':
        batch_indices = [batch_index for state in states for batch_index in state.batch_indices]
        action_histories = [action_history for state in states for action_history in state.action_history]
        scores = [score for state in states for score in state.score]
        rnn_states = [rnn_state for state in states for rnn_state in state.rnn_state]
        grammar_states = [grammar_state for state in states for grammar_state in state.grammar_state]
        if states[0].debug_info is not None:
            debug_info = [debug_info for state in states for debug_info in state.debug_info]
        else:
            debug_info = None
        return GrammarBasedState(batch_indices=batch_indices,
                                 action_history=action_histories,
                                 score=scores,
                                 rnn_state=rnn_states,
                                 grammar_state=grammar_states,
                                 possible_actions=states[0].possible_actions,
                                 extras=states[0].extras,
                                 debug_info=debug_info)
