import torch

from allennlp.common import Registrable


class Sampler(Registrable):
    """
    An abstract class representing a multinomial sampler
    """

    def __call__(
        self,
        log_probs: torch.Tensor,
        perturbed_log_probs: torch.Tensor = None,
        num_samples: int = 1,
    ) -> torch.Tensor:
        raise NotImplementedError
