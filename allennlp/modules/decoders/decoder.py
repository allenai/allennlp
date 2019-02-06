from typing import Dict
import torch

from allennlp.common.registrable import Registrable

class Decoder(torch.nn.Module, Registrable):
    """
    ``Decoder`` class is a wrapper for different decoders
    """
    def forward(self,  # pylint: disable=arguments-differ
                target_tokens: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
