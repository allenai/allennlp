from typing import Generator, Generic, Set, TypeVar

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
                  allowed_actions: Set = None) -> Generator[StateType, None, None]:
        """
        The main method in the ``DecoderStep`` API.  This function defines the computation done at
        each step of decoding and yields a ranked list of next states.

        Parameters
        ----------
        state : ``DecoderState``
            The current state of the decoder, which we will take a step `from`.
        allowed_actions : ``Set``
            If the ``DecoderTrainer`` has constraints on which actions need to be evaluated (e.g.,
            maximum marginal likelihood only needs to evaluate action sequences in a given set),
            you can pass those constraints here, to avoid constructing state objects unnecessarily.

        Returns
        -------
        next_states : ``Generator[DecoderState, None, None]``
            A generator for next states, ordered by score, which the decoding algorithm can sample
            from as it wishes.
        """
        raise NotImplementedError
