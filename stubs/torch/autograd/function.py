# TODO(joelgrus): do better than this
from typing import Any

import torch

class Function:
    def __call__(self, *args, **kwargs) -> Any: ...

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...

    @staticmethod
    def forward(ctx, *args, **kwargs) -> Any: ...

    @staticmethod
    def backward(ctx, *grad_outputs) -> Any: ...
