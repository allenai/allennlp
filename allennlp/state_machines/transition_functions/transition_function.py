from typing import Generic, List, Set, TypeVar

import torch

from allennlp.state_machines.states import State

StateType = TypeVar('StateType', bound=State)  # pylint: disable=invalid-name


class TransitionFunction(torch.nn.Module, Generic[StateType]):
    """
    A ``TransitionFunction`` is a module that assigns scores to state transitions in a
    transition-based decoder.

    The ``TransitionFunction`` takes a ``State`` and outputs a ranked list of next states, ordered
    by the state's score.

    The intention with this class is that a model will implement a subclass of
    ``TransitionFunction`` that defines how exactly you want to handle the input and what
    computations get done at each step of decoding, and how states are scored.  This subclass then
    gets passed to a ``DecoderTrainer`` to have its parameters trained.
    """
    def take_step(self,
                  state: StateType,
                  max_actions: int = None,
                  allowed_actions: List[Set] = None) -> List[StateType]:
        """
        The main method in the ``TransitionFunction`` API.  This function defines the computation
        done at each step of decoding and returns a ranked list of next states.

        The input state is `grouped`, to allow for efficient computation, but the output states
        should all have a ``group_size`` of 1, to make things easier on the decoding algorithm.
        They will get regrouped later as needed.

        Because of the way we handle grouping in the decoder states, constructing a new state is
        actually a relatively expensive operation.  If you know a priori that only some of the
        states will be needed (either because you have a set of gold action sequences, or you have
        a fixed beam size), passing that information into this function will keep us from
        constructing more states than we need, which will greatly speed up your computation.

        IMPORTANT: This method `must` returns states already sorted by their score, otherwise
        ``BeamSearch`` and other methods will break.  For efficiency, we do not perform an
        additional sort in those methods.

        ALSO IMPORTANT: When ``allowed_actions`` is given and ``max_actions`` is not, we assume you
        want to evaluate all possible states and do not need any sorting (e.g., this is true for
        maximum marginal likelihood training that does not use a beam search).  In this case, we
        may skip the sorting step for efficiency reasons.

        Parameters
        ----------
        state : ``State``
            The current state of the decoder, which we will take a step `from`.  We may be grouping
            together computation for several states here.  Because we can have several states for
            each instance in the original batch being evaluated at the same time, we use
            ``group_size`` for this kind of batching, and ``batch_size`` for the `original` batch
            in ``model.forward.``
        max_actions : ``int``, optional
            If you know that you will only need a certain number of states out of this (e.g., in a
            beam search), you can pass in the max number of actions that you need, and we will only
            construct that many states (for each `batch` instance - `not` for each `group`
            instance!).  This can save a whole lot of computation if you have an action space
            that's much larger than your beam size.
        allowed_actions : ``List[Set]``, optional
            If the ``DecoderTrainer`` has constraints on which actions need to be evaluated (e.g.,
            maximum marginal likelihood only needs to evaluate action sequences in a given set),
            you can pass those constraints here, to avoid constructing state objects unnecessarily.
            If there are no constraints from the trainer, passing a value of ``None`` here will
            allow all actions to be considered.

            This is a list because it is `batched` - every instance in the batch has a set of
            allowed actions.  Note that the size of this list is the ``group_size`` in the
            ``State``, `not` the ``batch_size`` of ``model.forward``.  The training algorithm needs
            to convert from the `batched` allowed action sequences that it has to a `grouped`
            allowed action sequence list.

        Returns
        -------
        next_states : ``List[State]``
            A list of next states, ordered by score.
        """
        raise NotImplementedError
