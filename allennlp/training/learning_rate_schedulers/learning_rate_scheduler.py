from typing import Any, Dict, List, Union

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.registrable import Registrable
from allennlp.training.scheduler import Scheduler
from allennlp.training.optimizers import Optimizer


class LearningRateScheduler(Scheduler, Registrable):
    def __init__(self, optimizer: torch.optim.Optimizer, last_epoch: int = -1) -> None:
        super().__init__(optimizer, "lr", last_epoch)

    @overrides
    def get_values(self):
        raise NotImplementedError


class _PyTorchLearningRateSchedulerWrapper(LearningRateScheduler):
    def __init__(self, lr_scheduler: torch.optim.lr_scheduler._LRScheduler) -> None:
        self.lr_scheduler = lr_scheduler

    def get_values(self):
        return self.lr_scheduler.get_last_lr()

    @overrides
    def step(self, metric: float = None) -> None:
        self.lr_scheduler.step()

    @overrides
    def state_dict(self) -> Dict[str, Any]:
        return self.lr_scheduler.state_dict()

    @overrides
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.lr_scheduler.load_state_dict(state_dict)


class _PyTorchLearningRateSchedulerWithMetricsWrapper(_PyTorchLearningRateSchedulerWrapper):
    @overrides
    def step(self, metric: float = None) -> None:
        if metric is None:
            raise ConfigurationError(
                "This learning rate scheduler requires "
                "a validation metric to compute the schedule and therefore "
                "must be used with a validation dataset."
            )
        self.lr_scheduler.step(metric)


@LearningRateScheduler.register("step")
class StepLearningRateScheduler(_PyTorchLearningRateSchedulerWrapper):
    """
    Registered as a `LearningRateScheduler` with name "step".
    """

    def __init__(
        self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1
    ) -> None:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch
        )
        super().__init__(lr_scheduler)


@LearningRateScheduler.register("multi_step")
class MultiStepLearningRateScheduler(_PyTorchLearningRateSchedulerWrapper):
    """
    Registered as a `LearningRateScheduler` with name "multi_step".
    """

    def __init__(
        self, optimizer: Optimizer, milestones: List[int], gamma: float = 0.1, last_epoch: int = -1
    ) -> None:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=milestones, gamma=gamma, last_epoch=last_epoch
        )
        super().__init__(lr_scheduler)


@LearningRateScheduler.register("exponential")
class ExponentialLearningRateScheduler(_PyTorchLearningRateSchedulerWrapper):
    """
    Registered as a `LearningRateScheduler` with name "exponential".
    """

    def __init__(self, optimizer: Optimizer, gamma: float = 0.1, last_epoch: int = -1) -> None:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=gamma, last_epoch=last_epoch
        )
        super().__init__(lr_scheduler)


@LearningRateScheduler.register("reduce_on_plateau")
class ReduceOnPlateauLearningRateScheduler(_PyTorchLearningRateSchedulerWithMetricsWrapper):
    """
    Registered as a `LearningRateScheduler` with name "reduce_on_plateau".
    """

    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        verbose: bool = False,
        threshold_mode: str = "rel",
        threshold: float = 1e-4,
        cooldown: int = 0,
        min_lr: Union[float, List[float]] = 0,
        eps: float = 1e-8,
    ) -> None:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            verbose=verbose,
            threshold=threshold,
            cooldown=cooldown,
            min_lr=min_lr,
        )
        super().__init__(lr_scheduler)
