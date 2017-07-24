import torch

from allennlp.experiments.registry import Registry
from allennlp.training.regularizer import Regularizer


@Registry.register_regularizer("l1")
class L1Regularizer(Regularizer):

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha

    def __call__(self, parameter: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.sum(torch.abs(parameter))


@Registry.register_regularizer("l2")
class L2Regularizer(Regularizer):

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha

    def __call__(self, parameter: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.sum(torch.pow(parameter, 2))
