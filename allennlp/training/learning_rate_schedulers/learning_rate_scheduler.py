from typing import Dict, Any, Optional

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.registrable import Registrable
from allennlp.training.scheduler import Scheduler


class LearningRateScheduler(Scheduler, Registrable):
    def __init__(self, optimizer: torch.optim.Optimizer, last_epoch: int = -1) -> None:
        super().__init__(optimizer, "lr", last_epoch)

    def get_values(self) -> None:
        raise NotImplementedError

    # Requires custom from_params so we can wrap the PyTorch LR schedulers.
    @classmethod
    def from_params(
        cls,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        batches_per_epoch: Optional[int],
        params: Params,
    ):  # type: ignore
        scheduler_type = params.pop_choice("type", LearningRateScheduler.list_available())
        constructor = LearningRateScheduler.by_name(scheduler_type)
        from inspect import signature

        constructor_parameters = signature(constructor).parameters.keys()
        params = params.as_dict()
        if "num_epochs" in constructor_parameters and "num_epochs" not in params:
            params["num_epochs"] = num_epochs
        if "batches_per_epoch" in constructor_parameters and "batches_per_epoch" not in params:
            params["batches_per_epoch"] = batches_per_epoch
        scheduler = constructor(optimizer, **params)  # type: ignore
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return _PyTorchLearningRateSchedulerWithMetricsWrapper(scheduler)
        elif isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
            return _PyTorchLearningRateSchedulerWrapper(scheduler)
        else:
            return scheduler


class _PyTorchLearningRateSchedulerWrapper(LearningRateScheduler):
    def __init__(self, lr_scheduler: torch.optim.lr_scheduler._LRScheduler) -> None:
        self.lr_scheduler = lr_scheduler

    def get_values(self):
        return self.lr_scheduler.get_lr()

    @overrides
    def step(self, metric: float = None, epoch: int = None) -> None:
        self.lr_scheduler.step(epoch)

    @overrides
    def state_dict(self) -> Dict[str, Any]:
        return self.lr_scheduler.state_dict()

    @overrides
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.lr_scheduler.load_state_dict(state_dict)


class _PyTorchLearningRateSchedulerWithMetricsWrapper(_PyTorchLearningRateSchedulerWrapper):
    @overrides
    def step(self, metric: float = None, epoch: int = None) -> None:
        if metric is None:
            raise ConfigurationError(
                "This learning rate scheduler requires "
                "a validation metric to compute the schedule and therefore "
                "must be used with a validation dataset."
            )
        self.lr_scheduler.step(metric, epoch)


# Force PyTorch learning rate schedulers into the registry.
Registrable._registry[LearningRateScheduler] = {
    "step": (torch.optim.lr_scheduler.StepLR, None),
    "multi_step": (torch.optim.lr_scheduler.MultiStepLR, None),
    "exponential": (torch.optim.lr_scheduler.ExponentialLR, None),
    "reduce_on_plateau": (torch.optim.lr_scheduler.ReduceLROnPlateau, None),
}
