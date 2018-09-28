from typing import List, Sequence

import torch

from allennlp.state_machines.states.state import State
from allennlp.state_machines.states.rnn_statelet import RnnStatelet


class SimpleState(State["SimpleState"]):
    """
    A basic state that can be used with a seq2seq model, like 'simple_seq2seq'.

    Parameters
    ----------
    batch_indices : ``List[int]``
        Passed to super class; see docs there.
    action_history : ``List[List[int]]``
        Passed to super class; see docs there.
    score : ``List[torch.Tensor]``
        Passed to super class; see docs there.
    rnn_state : ``List[RnnStatelet]``
        An `RnnStatelet` for every group element.  This keeps track of the current decoder hidden
        state, the previous decoder output, the output from the encoder (for computing attentions),
        and other things that are typical seq2seq decoder state things.
    end_index : ``int``
        This is how we know a decoded sequence is complete.
    """

    def __init__(self,
                 batch_indices: List[int],
                 action_history: List[List[int]],
                 score: List[torch.Tensor],
                 rnn_state: List[RnnStatelet],
                 end_index: int) -> None:
        super().__init__(batch_indices, action_history, score)
        self.rnn_state = rnn_state
        self.end_index = end_index

    def is_finished(self) -> bool:
        if len(self.batch_indices) != 1:
            raise RuntimeError("is_finished() is only defined with a group_size of 1")
        return self.action_history[0][-1] == self.end_index

    @classmethod
    def combine_states(cls, states: Sequence["SimpleState"]) -> "SimpleState":
        batch_indices = [batch_index for state in states for batch_index in state.batch_indices]
        action_histories = [action_history for state in states for action_history in state.action_history]
        scores = [score for state in states for score in state.score]
        rnn_states = [rnn_state for state in states for rnn_state in state.rnn_state]
        end_index = states[0].end_index
        return cls(batch_indices, action_histories, scores, rnn_states, end_index)
