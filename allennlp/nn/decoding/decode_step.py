from typing import Generator, Set

import torch

from allennlp.nn.decoding.decoder_state import DecoderState


class DecodeStep(torch.nn.Module):
    def take_step(self,
                  state: DecoderState,
                  decoder_input: torch.Tensor,
                  allowed_actions: Set[int] = None,
                  max_next_states: int = None) -> Generator[DecoderState, None, None]:
        raise NotImplementedError
