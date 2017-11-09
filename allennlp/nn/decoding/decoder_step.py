from typing import Generator, Generic, List, Set, TypeVar

import torch

from allennlp.nn.decoding.decoder_state import DecoderState

StateType = TypeVar('StateType', bound=DecoderState)  # pylint: disable=invalid-name


class DecoderStep(torch.nn.Module, Generic[StateType]):
    """
    A ``DecoderStep`` is a module that assigns scores to state transitions in a transition-based
    decoder.

    The ``DecoderStep`` takes a ``DecoderState`` and outputs a ranked list of next states, ordered
    by the state's score.

    The intention with this class is that a model will implement a subclass of ``DecoderStep`` that
    defines how exactly you want to handle the input and what computations get done at each step of
    decoding, and how states are scored.  This subclass then gets passed to a ``DecoderTrainer`` to
    have its parameters trained.
    """
    def take_step(self,
                  state: StateType,
                  allowed_actions: List[Set] = None) -> Generator[StateType, None, None]:
        """
        The main method in the ``DecoderStep`` API.  This function defines the computation done at
        each step of decoding and yields a ranked list of next states.

        Parameters
        ----------
        state : ``DecoderState``
            The current state of the decoder, which we will take a step `from`.  We may be grouping
            together computation for several states here.  Because we can have several states for
            each instance in the original batch being evaluated at the same time, we use
            ``group_size`` for this kind of batching, and ``batch_size`` for the `original` batch
            in ``model.forward.``
        allowed_actions : ``List[Set]``
            If the ``DecoderTrainer`` has constraints on which actions need to be evaluated (e.g.,
            maximum marginal likelihood only needs to evaluate action sequences in a given set),
            you can pass those constraints here, to avoid constructing state objects unnecessarily.
            This is a list because it is `batched` - every instance in the batch has a set of
            allowed actions.  Note that the size of this list is the ``group_size`` in the
            ``DecoderState``, `not` the ``batch_size`` of ``model.forward``.  The training
            algorithm needs to convert from the `batched` allowed action sequences that it has to a
            `grouped` allowed action sequence list.

        Returns
        -------
        next_states : ``Generator[DecoderState, None, None]``
            A generator for next states, ordered by score, which the decoding algorithm can sample
            from as it wishes.
        """
        raise NotImplementedError
