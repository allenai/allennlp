from typing import Callable, Union, List

from torch.optim import Optimizer

EpochToFactor = Callable[[int], float]

class _LRScheduler:
    def get_lr(self) -> List[float]: ...

    def step(self, epoch: int = None) -> None: ...

class LambdaLR(_LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 lr_lambda: Union[EpochToFactor, List[EpochToFactor]],
                 last_epoch: int = -1) -> None: ...

class StepLR(_LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 step_size: int,
                 gamma: float = 0.1,
                 last_epoch: int = -1) -> None: ...

class MultiStepLR(_LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 milestones: List[int],
                 gamma: float = 0.1,
                 last_epoch: int = -1) -> None: ...

class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, gamma: float, last_epoch: int = -1) -> None: ...

class ReduceLROnPlateau(_LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 mode: str = 'min',
                 factor: float = 0.1,
                 patience: int = 10,
                 verbose: bool = False,
                 threshold: float = 1e-4,
                 threshold_mode: str = 'rel',
                 cooldown: int = 0,
                 min_lr: Union[float, List[float]] = 0,
                 eps: float = 1e-8) -> None: ...
