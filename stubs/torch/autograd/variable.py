from typing import Union, Sequence, Any, Optional, TypeVar, Generic, overload, Tuple, Sized, Callable

import torch
import numpy as np

Index = Union['Variable', torch.Tensor, int, Sequence[Union[int, slice]]]
Value = Union[float, int, 'Variable', torch.Tensor]

T = TypeVar('T', bound=torch._TensorBase)

Hook = Callable[['Variable'], Optional['Variable']]

class Variable(Generic[T], Sized):
    def __init__(self, data: T, requires_grad: bool = False, volatile: bool = False) -> None: ...

    data: T
    grad: T
    requires_grad: bool
    volatile: bool
    is_leaf: bool

    def register_hook(self, hook: Hook) -> Hook: ...

    def backward(self,
                 gradient: Union[T, 'Variable', None] = None,
                 retain_graph: bool = False,
                 create_graph: bool = False,
                 retain_variables: bool = False) -> 'Variable': ...

    def long(self) -> 'Variable': ...
    def dim(self) -> int: ...
    def ndim(self) -> int: ...

    def norm(self,
             p: float = 2.,
             dim: Optional[int] = None,
             keepdim: bool = False) -> 'Variable': ...

    @overload
    def size(self) -> Sequence[int]: ...
    @overload
    def size(self, dim: int) -> int: ...
    def size(self, dim: Optional[int] = None) -> Union[int, Sequence[int]]: ...

    def __getitem__(self, key: Index) -> 'Variable': ...
    def __setitem__(self, key: Index, value: Value) -> 'Variable': ...

    def __add__(self, other: Value) -> 'Variable': ...
    def __radd__(self, other: Value) -> 'Variable': ...
    def __iadd__(self, other: Value) -> 'Variable': ...

    def __sub__(self, other: Value) -> 'Variable': ...
    def __rsub__(self, other: Value) -> 'Variable': ...
    def __isub__(self, other: Value) -> 'Variable': ...

    def __mul__(self, other: Value) -> 'Variable': ...
    def __rmul__(self, other: Value) -> 'Variable': ...
    def __imul__(self, other: Value) -> 'Variable': ...
    def __matmul__(self, other: Value) -> 'Variable': ...

    def __truediv__(self, other: Value) -> 'Variable': ...
    def __rtruediv__(self, other: Value) -> 'Variable': ...
    def __itruediv__(self, other: Value) -> 'Variable': ...

    # see https://github.com/python/mypy/issues/2783#issuecomment-276596902
    def __eq__(self, other: Any) -> 'Variable': ...  # type: ignore
    def __lt__(self, other: Any) -> 'Variable': ...  # type: ignore
    def __le__(self, other: Any) -> 'Variable': ...  # type: ignore
    def __gt__(self, other: Any) -> 'Variable': ...  # type: ignore
    def __ge__(self, other: Any) -> 'Variable': ...  # type: ignore
    def __ne__(self, other: Any) -> 'Variable': ...  # type: ignore

    def __neg__(self) -> 'Variable': ...

    def div(self, other: Union[Variable, torch._TensorBase]) -> 'Variable': ...

    def max(self, dim: Optional[int] = None, keepdim: Optional[bool] = None) -> 'Variable': ...
    def squeeze(self, dim: Optional[int] = None) -> 'Variable': ...
    def unsqueeze(self, dim: int) -> 'Variable': ...

    def sort(self, dim: Optional[int] = None, descending: bool = False) -> Tuple['Variable', 'Variable']: ...
    def index_select(self, dim: int, index: Index) -> 'Variable': ...

    def view(self, *dims: int) -> 'Variable': ...
    def expand_as(self, other: 'Variable') -> 'Variable': ...
    def expand(self, *dims: int) -> 'Variable': ...
    def contiguous(self) -> 'Variable': ...

    def __len__(self) -> int: ...

    def clone(self) -> 'Variable': ...
    def float(self) -> 'Variable': ...

    def is_cuda(self) -> bool: ...
    def cuda(self, device: Optional[int] = None, async: bool = False) -> 'Variable': ...
    def cpu(self) -> 'Variable': ...

    def numpy(self) -> np.ndarray: ...

    def sum(self,
            dim: Optional[int] = None,
            keepdim: Optional[bool] = False,
            out: Optional[T] = None) -> 'Variable': ...

    def permute(self, *dims: int) -> 'Variable': ...

