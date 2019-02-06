from typing import Dict, Any

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.registrable import Registrable
from allennlp.training.scheduler import Scheduler


class LearningRateScheduler(Scheduler, Registrable):

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 last_epoch: int = -1) -> None:
        super().__init__(optimizer, "lr", last_epoch)

    def get_values(self) -> None:
        raise NotImplementedError

    # Requires custom from_params so we can wrap the PyTorch LR schedulers.
    @classmethod
    def from_params(cls, optimizer: torch.optim.Optimizer, params: Params):  # type: ignore
        # pylint: disable=arguments-differ
        scheduler_type = params.pop_choice("type", LearningRateScheduler.list_available())
        scheduler = LearningRateScheduler.by_name(scheduler_type)(optimizer, **params.as_dict())  # type: ignore
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return _PyTorchLearningRateSchedulerWithMetricsWrapper(scheduler)
        elif isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):  # pylint: disable=protected-access
            return _PyTorchLearningRateSchedulerWrapper(scheduler)
        else:
            return scheduler


class _PyTorchLearningRateSchedulerWrapper(LearningRateScheduler):

    def __init__(self, lr_scheduler: torch.optim.lr_scheduler._LRScheduler) -> None:  # pylint: disable=protected-access,super-init-not-called
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
            raise ConfigurationError("This learning rate scheduler requires "
                                     "a validation metric to compute the schedule and therefore "
                                     "must be used with a validation dataset.")
        self.lr_scheduler.step(metric, epoch)


# Force PyTorch learning rate schedulers into the registry.
Registrable._registry[LearningRateScheduler] = {   # pylint: disable=protected-access
        "step": torch.optim.lr_scheduler.StepLR,
        "multi_step": torch.optim.lr_scheduler.MultiStepLR,
        "exponential": torch.optim.lr_scheduler.ExponentialLR,
        "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
}
