from typing import Dict, Generic, TypeVar

import torch

from allennlp.nn.decoding.decoder_step import DecoderStep
from allennlp.nn.decoding.decoder_state import DecoderState

SupervisionType = TypeVar('SupervisionType')

class DecoderTrainer(Generic[SupervisionType]):
    """
    ``DecoderTrainers`` define a training regime for transition-based decoders.  A
    ``DecoderTrainer`` assumes an initial ``DecoderState``, a ``DecoderStep`` function that can
    traverse the state space, and some representation of target or gold action sequences.  Given
    these things, the ``DecoderTrainer`` trains the ``DecoderStep`` function to traverse the state
    space to end up at good end states.

    Concrete implementations of this abstract base class could do things like maximum marginal
    likelihood, SEARN, LaSO, or other structured learning algorithms.  If you're just trying to
    maximize the probability of a single target sequence, there are way more efficient ways to do
    that than using this API.
    """
    # TODO(mattg): Make `DecoderTrainer` generic over supervision type, so we can either take a
    # tensor of targets, or a reward function, or something else.
    def decode(self,
               initial_state: DecoderState,
               decode_step: DecoderStep,
               supervision: SupervisionType) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
