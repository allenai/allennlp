from typing import Iterable, Callable, Optional, Tuple

class Optimizer:
    def load_state_dict(state_dict: dict) -> None: ...
    def state_dict() -> dict: ...
    def step(closure: Optional[Callable] = None) -> None: ...
    def zero_grad() -> None: ...

class Adadelta(Optimizer):
    def __init__(self,
                 params: Iterable,
                 lr: float = 1.0,
                 rho: float = 0.9,
                 eps: float = 1e-06,
                 weight_decay: float = 0) -> None: ...

class Adagrad(Optimizer):
    def __init__(self,
                 params: Iterable,
                 lr: float = 0.01,
                 lr_decay: float = 0,
                 weight_decay: float = 0) -> None: ...

class Adam(Optimizer):
    def __init__(self,
                 params: Iterable,
                 lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-08,
                 weight_decay: float = 0) -> None: ...

class RMSprop(Optimizer):
    def __init__(self,
                 params: Iterable,
                 lr: float = 0.01,
                 alpha: float = 0.99,
                 eps: float = 1e-08,
                 weight_decay: float = 0,
                 momentum: float = 0,
                 centered: bool = False) -> None: ...

class SGD(Optimizer):
    def __init__(self,
                 params: Iterable,
                 lr: float,
                 momentum: float = 0,
                 dampening: float =0,
                 weight_decay: float = 0,
                 nesterov: bool = False) -> None: ...
