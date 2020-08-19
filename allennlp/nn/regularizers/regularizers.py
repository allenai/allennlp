import torch

from allennlp.nn.regularizers.regularizer import Regularizer


@Regularizer.register("l1")
class L1Regularizer(Regularizer):
    """
    Represents a penalty proportional to the sum of the absolute values of the parameters

    Registered as a `Regularizer` with name "l1".
    """

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha

    def __call__(self, parameter: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.sum(torch.abs(parameter))


@Regularizer.register("l2")
class L2Regularizer(Regularizer):
    """
    Represents a penalty proportional to the sum of squared values of the parameters

    Registered as a `Regularizer` with name "l2".
    """

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha

    def __call__(self, parameter: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.sum(torch.pow(parameter, 2))
