from typing import Generic, List, TypeVar

import torch

# Note that the bound here is `State` itself.  This is what lets us have methods that take
# lists of a `State` subclass and output structures with the subclass.  Really ugly that we
# have to do this generic typing _for our own class_, but it makes mypy happy and gives us good
# type checking in a few important methods.
T = TypeVar('T', bound='State')

class State(Generic[T]):
    """
    Represents the (batched) state of a transition-based decoder.

    There are two different kinds of batching we need to distinguish here.  First, there's the
    batch of training instances passed to ``model.forward()``.  We'll use "batch" and
    ``batch_size`` to refer to this through the docs and code.  We additionally batch together
    computation for several states at the same time, where each state could be from the same
    training instance in the original batch, or different instances.  We use "group" and
    ``group_size`` in the docs and code to refer to this kind of batching, to distinguish it from
    the batch of training instances.

    So, using this terminology, a single ``State`` object represents a `grouped` collection of
    states.  Because different states in this group might finish at different timesteps, we have
    methods and member variables to handle some bookkeeping around this, to split and regroup
    things.

    Parameters
    ----------
    batch_indices : ``List[int]``
        A ``group_size``-length list, where each element specifies which ``batch_index`` that group
        element came from.

        Our internal variables (like scores, action histories, hidden states, whatever) are
        `grouped`, and our ``group_size`` is likely different from the original ``batch_size``.
        This variable keeps track of which batch instance each group element came from (e.g., to
        know what the correct action sequences are, or which encoder outputs to use).
    action_history : ``List[List[int]]``
        The list of actions taken so far in this state.  This is also grouped, so each state in the
        group has a list of actions.
    score : ``List[torch.Tensor]``
        This state's score.  It's a variable, because typically we'll be computing a loss based on
        this score, and using it for backprop during training.  Like the other variables here, this
        is a ``group_size``-length list.
    """
    def __init__(self,
                 batch_indices: List[int],
                 action_history: List[List[int]],
                 score: List[torch.Tensor]) -> None:
        self.batch_indices = batch_indices
        self.action_history = action_history
        self.score = score

    def is_finished(self) -> bool:
        """
        If this state has a ``group_size`` of 1, this returns whether the single action sequence in
        this state is finished or not.  If this state has a ``group_size`` other than 1, this
        method raises an error.
        """
        raise NotImplementedError

    @classmethod
    def combine_states(cls, states: List[T]) -> T:
        """
        Combines a list of states, each with their own group size, into a single state.
        """
        raise NotImplementedError

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented
