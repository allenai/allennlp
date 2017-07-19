import torch


class Regularizer:
    """
    An abstract class representing a regularizer. It must implement
    call, returning a scalar tensor.
    """

    def __init__(self, **kwargs):
        raise NotImplementedError

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
