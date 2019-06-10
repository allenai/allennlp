from typing import Dict

import torch
from torch.nn import Module

from allennlp.common import Registrable
from allennlp.modules import Embedding


class SeqDecoder(Module, Registrable):
    """
    A ``SeqDecoder`` is a base class for different types of Seq decoding modules
    Parameters
    ----------
    target_embedder : ``Embedding``
        Embedder for target tokens. Needed in the base class to enable weight tying.
    """
    default_implementation = 'auto_regressive_seq_decoder'

    def __init__(self,
                 target_embedder: Embedding) -> None:
        super().__init__()
        self.target_embedder = target_embedder

    def get_output_dim(self):
        raise NotImplementedError()

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        raise NotImplementedError()

    def forward(self,
                encoder_out: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def post_process(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()
