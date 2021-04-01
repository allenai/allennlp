from typing import Any, Dict, List, Union

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.registrable import Registrable
from allennlp.training.scheduler import Scheduler
from allennlp.training.optimizers import Optimizer

from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)


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
    Registered as a `LearningRateScheduler` with name "step".  The "optimizer" argument does not get
    an entry in a configuration file for the object.
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
    Registered as a `LearningRateScheduler` with name "multi_step".  The "optimizer" argument does
    not get an entry in a configuration file for the object.
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
    Registered as a `LearningRateScheduler` with name "exponential".  The "optimizer" argument does
    not get an entry in a configuration file for the object.
    """

    def __init__(self, optimizer: Optimizer, gamma: float = 0.1, last_epoch: int = -1) -> None:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=gamma, last_epoch=last_epoch
        )
        super().__init__(lr_scheduler)


@LearningRateScheduler.register("reduce_on_plateau")
class ReduceOnPlateauLearningRateScheduler(_PyTorchLearningRateSchedulerWithMetricsWrapper):
    """
    Registered as a `LearningRateScheduler` with name "reduce_on_plateau".  The "optimizer" argument
    does not get an entry in a configuration file for the object.
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
            threshold_mode=threshold_mode,
            threshold=threshold,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
        )
        super().__init__(lr_scheduler)


@LearningRateScheduler.register("constant")
class ConstantLearningRateScheduler(_PyTorchLearningRateSchedulerWrapper):
    """
    Registered as a `LearningRateScheduler` with name "constant".  The "optimizer" argument does
    not get an entry in a configuration file for the object.
    """

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        lr_scheduler = get_constant_schedule(optimizer=optimizer, last_epoch=last_epoch)
        super().__init__(lr_scheduler)


@LearningRateScheduler.register("constant_with_warmup")
class ConstantWithWarmupLearningRateScheduler(_PyTorchLearningRateSchedulerWrapper):
    """
    Registered as a `LearningRateScheduler` with name "constant_with_warmup".  The "optimizer" argument does
    not get an entry in a configuration file for the object.
    """

    def __init__(self, optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1) -> None:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=num_warmup_steps, last_epoch=last_epoch
        )
        super().__init__(lr_scheduler)


@LearningRateScheduler.register("cosine_with_warmup")
class CosineWithWarmupLearningRateScheduler(_PyTorchLearningRateSchedulerWrapper):
    """
    Registered as a `LearningRateScheduler` with name "cosine_with_warmup".  The "optimizer" argument does
    not get an entry in a configuration file for the object.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ) -> None:
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            last_epoch=last_epoch,
        )
        super().__init__(lr_scheduler)


@LearningRateScheduler.register("cosine_hard_restarts_with_warmup")
class CosineHardRestartsWithWarmupLearningRateScheduler(_PyTorchLearningRateSchedulerWrapper):
    """
    Registered as a `LearningRateScheduler` with name "cosine_hard_restarts_with_warmup".
    The "optimizer" argument does not get an entry in a configuration file for the object.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: int = 1,
        last_epoch: int = -1,
    ) -> None:
        lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            last_epoch=last_epoch,
        )
        super().__init__(lr_scheduler)
