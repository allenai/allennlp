from typing import Optional, List, Iterator, Iterable

import torch

from .module import Module

class Parameter:
    pass

class ModuleList(Module, Iterable[Module]):
    def __init__(self, modules: List[Module]) -> None: ...

    def __iter__(self) -> Iterator[Module]: ...

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

class LSTM(Module): pass
class GRU(Module): pass
class RNN(Module): pass
