from typing import Dict

import torch

from allennlp.common import Params, Registrable
from allennlp.nn.decoding.decoder_step import DecoderStep
from allennlp.nn.decoding.decoder_state import DecoderState


class DecoderTrainer(Registrable):
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
    # TODO(mattg): figure out how reward functions fit into this.  We could _either_ take a targets
    # tensor _or_ a reward function over states?  Not really sure...
    def decode(self,
               initial_state: DecoderState,
               decode_step: DecoderStep,
               targets: torch.Tensor = None,
               target_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params) -> 'DecoderTrainer':
        choice = params.pop_choice('type', cls.list_available())
        return cls.by_name(choice).from_params(params)
