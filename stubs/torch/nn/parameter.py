from typing import TypeVar, Generic

import torch
from torch.autograd import Variable

T = TypeVar('T', bound=torch._TensorBase)

class Parameter(Variable, Generic[T]):
    def __init__(self, tensor: T, requires_grad: bool = False) -> None: ...

    data: T
