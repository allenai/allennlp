from typing import Optional, List, Iterator, Iterable, TypeVar, Generic, Union, Sequence

import torch

from .module import Module
from .modules import RNNBase
from .parameter import Parameter

# Modules
M = TypeVar('M', bound=Module)

class ModuleList(Module, Generic[M], Iterable[M]):
    def __init__(self, modules: List[M]) -> None: ...

    def __iter__(self) -> Iterator[M]: ...

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None: ...

    weight: Parameter
    bias: Optional[Parameter]

class Dropout(Module):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None: ...

class CrossEntropyLoss(Module):
    def __init__(self,
                 weight: Optional[torch.Tensor] = None,
                 size_average: Optional[bool] = None,
                 ignore_index: Optional[int] = None) -> None: ...

# RNNs
class LSTM(RNNBase): pass
class GRU(RNNBase): pass
class RNN(RNNBase): pass

Ints = Union[int, Sequence[int]]

# Convolutions
class Conv1d(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Ints,
                 stride: Ints = 1,
                 padding: Ints = 0,
                 dilation: Ints = 1,
                 groups: int = 1,
                 bias: bool = True) -> None: ...


# Activations
class Threshold(Module): ...

class ReLU(Threshold):
    def __init__(self, inplace: bool = False) -> None: ...

class RReLU(Module):
    def __init__(self, lower: float = 1. / 8, upper: float = 1. / 3, inplace: bool = False) -> None: ...

class Hardtanh(Module):
    def __init__(self,
                 min_val: float = -1,
                 max_val: float = 1,
                 inplace: bool = False,
                 min_value: None = None,
                 max_value: None = None) -> None: ...

class ReLU6(Hardtanh):
    def __init__(self, inplace: bool = False) -> None: ...

class Sigmoid(Module): ...

class Tanh(Module): ...

class ELU(Module):
    def __init__(self, alpha: float = 1., inplace: bool = False) -> None: ...

class SELU(Module):
    def __init__(self, inplace: bool = False) -> None: ...

class GLU(Module):
    def __init__(self, dim: int = -1) -> None: ...

class Hardshrink(Module):
    def __init__(self, lambd: float = 0.5) -> None: ...

class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 1e-2, inplace: bool = False) -> None: ...

class LogSigmoid(Module): ...

class Softplus(Module):
    def __init__(self, beta: float = 1, threshold: float = 20) -> None: ...

class Softshrink(Module):
    def __init__(self, lambd: float = 0.5) -> None: ...

class PReLU(Module):
    def __init__(self, num_parameters: int = 1, init: float = 0.25) -> None: ...

class Softsign(Module): ...

class Tanhshrink(Module): ...

class Softmin(Module): ...

class Softmax(Module): ...

class Softmax2d(Module): ...

class LogSoftmax(Module): ...
