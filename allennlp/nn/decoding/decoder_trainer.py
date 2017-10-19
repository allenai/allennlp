from typing import Callable, Dict, List, Tuple

import torch
from torch.autograd import Variable

from allennlp.common import Params, Registrable
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers.seq2seq import START_SYMBOL, END_SYMBOL
from allennlp.nn.decoding.decode_step import DecodeStep
from allennlp.nn.decoding.decoder_state import DecoderState
from allennlp.nn import util


class DecoderTrainer(Registrable):
    """
    ``DecoderTrainers`` define a training regime for transition-based decoders.  A
    ``DecoderTrainer`` assumes an initial ``DecoderState``, a ``DecodeStep`` function that can
    traverse the state space, and some representation of target or gold action sequences.  Given
    these things, the ``DecoderTrainer`` trains the ``DecodeStep`` function to traverse the state
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
               decode_step: DecodeStep,
               targets: torch.Tensor = None,
               target_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params) -> 'DecoderTrainer':
        choice = params.pop_choice('type', cls.list_available())
        return cls.by_name(choice).from_params(params)
