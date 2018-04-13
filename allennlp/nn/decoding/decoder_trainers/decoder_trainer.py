from typing import Dict, Generic, TypeVar

import torch

from allennlp.nn.decoding.decoder_step import DecoderStep
from allennlp.nn.decoding.decoder_state import DecoderState

SupervisionType = TypeVar('SupervisionType')  # pylint: disable=invalid-name

class DecoderTrainer(Generic[SupervisionType]):
    """
    ``DecoderTrainers`` define a training regime for transition-based decoders.  A
    ``DecoderTrainer`` assumes an initial ``DecoderState``, a ``DecoderStep`` function that can
    traverse the state space, and some supervision signal.  Given these things, the
    ``DecoderTrainer`` trains the ``DecoderStep`` function to traverse the state space to end up at
    good end states.

    Concrete implementations of this abstract base class could do things like maximum marginal
    likelihood, SEARN, LaSO, or other structured learning algorithms.  If you're just trying to
    maximize the probability of a single target sequence where the possible outputs are the same
    for each timestep (as in, e.g., typical machine translation training regimes), there are way
    more efficient ways to do that than using this API.
    """
    def decode(self,
               initial_state: DecoderState,
               decode_step: DecoderStep,
               supervision: SupervisionType) -> Dict[str, torch.Tensor]:
        """
        Takes an initial state object, a means of transitioning from state to state, and a
        supervision signal, and uses the supervision to train the transition function to pick
        "good" states.

        This function should typically return a ``loss`` key during training, which the ``Model``
        will use as its loss.

        Parameters
        ----------
        initial_state : ``DecoderState``
            This is the initial state for decoding, typically initialized after running some kind
            of encoder on some inputs.
        decode_step : ``DecoderStep``
            This is the transition function that scores all possible actions that can be taken in a
            given state, and returns a ranked list of next states at each step of decoding.
        supervision : ``SupervisionType``
            This is the supervision that is used to train the ``decode_step`` function to pick
            "good" states.  You can use whatever kind of supervision you want (e.g., a single
            "gold" action sequence, a set of possible "gold" action sequences, a reward function,
            etc.).  We use ``typing.Generics`` to make sure that our static type checker is happy
            with how you've matched the supervision that you provide in the model to the
            ``DecoderTrainer`` that you want to use.
        """
        raise NotImplementedError
