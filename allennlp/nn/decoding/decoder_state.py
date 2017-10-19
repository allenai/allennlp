from typing import List, Tuple

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

    def initial_input(self):
        """
        The input symbol or token that you would expect the first step of decoding to receive.  We
        need this so that the decoder trainer doesn't have to know the particulars of how you
        define start symbols.

        This is likely constant across all instances of a particular subclass, for but ease of
        implementation, we're making it an instance method, not a class method.
        """
        raise NotImplementedError

    def is_finished(self):
        """
        Is this an end state?  Often this will correspond to having the last action be an end
        symbol, but again, we're using this here so that the decoder trainer doesn't have to know
        the details of how your model defines end symbol.s
        """
        raise NotImplementedError
