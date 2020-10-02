import torch

from allennlp.common import Registrable


class Sampler(Registrable):
    """
    An abstract class representing a multinomial sampler
    """

    def __call__(self, parameter: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError