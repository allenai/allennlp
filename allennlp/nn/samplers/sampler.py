import torch

from allennlp.common import Registrable


class Sampler(Registrable):
    """
    An abstract class representing a multinomial sampler
    """

    def __call__(
        self, logits: torch.Tensor, num_samples: int = 1, with_replacement: bool = True
    ) -> torch.Tensor:
        raise NotImplementedError
