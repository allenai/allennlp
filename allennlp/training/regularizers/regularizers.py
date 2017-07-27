import torch

from allennlp.training.regularizer import Regularizer


@Regularizer.register("l1")
class L1Regularizer(Regularizer):

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha

    def __call__(self, parameter: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.sum(torch.abs(parameter))


@Regularizer.register("l2")
class L2Regularizer(Regularizer):

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha

    def __call__(self, parameter: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.sum(torch.pow(parameter, 2))
