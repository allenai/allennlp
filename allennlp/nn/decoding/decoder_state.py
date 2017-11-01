from typing import List

import torch


class DecoderState:
    """
    Represents the state of a transition-based decoder.

    Parameters
    ----------
    action_history : ``List[int]``
        The list of actions taken so far in this state.

        The type annotation says this is an ``int``, but none of the training logic relies on this
        being an ``int``.  In some cases, items from this list will get passed as inputs to
        ``DecodeStep``, so this must return items that are compatible with inputs to your
        ``DecodeStep`` class.
    score : ``torch.autograd.Variable``
        This state's score.  It's a variable, because typically we'll be computing a loss based on
        this score, and using it for backprop during training.
    """
    def __init__(self,
                 action_history: List[int],
                 score: torch.autograd.Variable) -> None:
        self.action_history = action_history
        self.score = score

    def is_finished(self) -> bool:
        """
        Is this an end state?  Often this will correspond to having the last action be an end
        symbol.
        """
        raise NotImplementedError
