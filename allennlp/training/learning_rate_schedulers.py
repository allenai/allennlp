"""
AllenNLP just uses
`PyTorch learning rate schedulers <http://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate>`_,
with a thin wrapper to allow registering them and instantiating them ``from_params``.

The available learning rate schedulers are

* `"step" <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.StepLR>`_
* `"multi_step" <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.MultiStepLR>`_
* `"exponential" <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.ExponentialLR>`_
* `"reduce_on_plateau" <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau>`_
* `"cosine" <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.CosineAnnealingLR>`_
"""

import logging
from typing import Optional

import numpy as np
import torch.optim.lr_scheduler
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.registrable import Registrable


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class LearningRateScheduler(Registrable):
    """
    This class just allows us to implement ``Registrable`` for Pytorch :class:`LRSchedulers`.
    """
    def __init__(self, lr_scheduler) -> None:
        self.lr_scheduler = lr_scheduler

    def step(self, metric: float, epoch: Optional[int] = None):
        raise NotImplementedError

    def step_batch(self, batch_num_total: Optional[int]):
        if batch_num_total is not None:
            if hasattr(self.lr_scheduler, 'step_batch'):
                self.lr_scheduler.step_batch(batch_num_total)
            return

    # Requires custom from_params
    @classmethod
    def from_params(cls, optimizer: torch.optim.Optimizer, params: Params):  # type: ignore
        # pylint: disable=arguments-differ
        scheduler = params.pop_choice("type", LearningRateScheduler.list_available())

        schedulers = LearningRateScheduler.by_name(scheduler)(optimizer, **params.as_dict())  # type: ignore
        if isinstance(schedulers, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return LearningRateWithMetricsWrapper(schedulers)
        else:
            return LearningRateWithoutMetricsWrapper(schedulers)


class LearningRateWithoutMetricsWrapper(LearningRateScheduler):
    """
    A wrapper around learning rate schedulers that do not require metrics
    """
    def __init__(self, lr_scheduler: torch.optim.lr_scheduler._LRScheduler) -> None:  # pylint: disable=protected-access
        super().__init__(lr_scheduler)
        self.lr_scheduler = lr_scheduler

    @overrides
    def step(self, metric: float, epoch: Optional[int] = None):
        self.lr_scheduler.step(epoch)


class LearningRateWithMetricsWrapper(LearningRateScheduler):
    """
    A wrapper around learning rate schedulers that require metrics,
    At the moment there is only a single instance of this lrs. It is the ReduceLROnPlateau
    """
    def __init__(self, lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau) -> None:
        super().__init__(lr_scheduler)
        self.lr_scheduler = lr_scheduler

    @overrides
    def step(self, metric: float, epoch: Optional[int] = None):
        if metric is None:
            raise ConfigurationError("The reduce_on_plateau learning rate scheduler requires "
                                     "a validation metric to compute the schedule and therefore "
                                     "must be used with a validation dataset.")
        self.lr_scheduler.step(metric, epoch)


class NoamLR(torch.optim.lr_scheduler._LRScheduler): # pylint: disable=protected-access
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.

    Parameters
    ----------
    model_size : ``int``, required.
        The hidden size parameter which dominates the number of parameters in your model.
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    factor : ``float``, optional (default = 1.0).
        The overall scale factor for the learning rate decay.
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 model_size: int,
                 warmup_steps: int,
                 factor: float = 1.0,
                 last_epoch: int = -1) -> None:
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.model_size = model_size
        super().__init__(optimizer, last_epoch=last_epoch)

    def step(self, epoch=None):
        pass

    def step_batch(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, learning_rate in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = learning_rate

    def get_lr(self):
        step = max(self.last_epoch, 1)
        scale = self.factor *  (self.model_size ** (-0.5) *
                                min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))

        return [scale for _ in range(len(self.base_lrs))]


class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):  # pylint: disable=protected-access
    """
    Cosine annealing with restarts.

    This is decribed in the paper https://arxiv.org/abs/1608.03983.

    Parameters
    ----------
    optimizer : ``torch.optim.Optimizer``

    t_max : ``int``
        The maximum number of iterations within the first cycle.

    eta_min : ``float``, optional (default=0)
        The minimum learning rate.

    last_epoch : ``int``, optional (default=-1)
        The index of the last epoch. This is used when restarting.

    factor : ``float``, optional (default=1)
        The factor by which the cycle length (``T_max``) increases after each restart.

    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_max: int,
                 eta_min: float = 0.,
                 last_epoch: int = -1,
                 factor: float = 1.) -> None:
        assert t_max > 0
        assert eta_min >= 0
        if t_max == 1 and factor == 1:
            logger.warning("Cosine annealing scheduler will have no effect on the learning "
                           "rate since T_max = 1 and factor = 1.")
        self.t_max = t_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart: int = 0
        self._cycle_counter: int = 0
        self._cycle_factor: float = 1.
        self._updated_cycle_len: int = t_max
        self._initialized: bool = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time ``self.get_lr()`` was called,
        # since ``torch.optim.lr_scheduler._LRScheduler`` will call ``self.get_lr()``
        # when first initialized, but the learning rate should remain unchanged
        # for the first epoch.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
                self.eta_min + ((lr - self.eta_min) / 2) * (
                        np.cos(
                                np.pi *
                                (self._cycle_counter % self._updated_cycle_len) /
                                self._updated_cycle_len
                        ) + 1
                )
                for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.t_max)
            self._last_restart = step

        return lrs


# We just use the Pytorch LRSchedulers, so here we force them into
# Registry._registry so we can build them from params.
Registrable._registry[LearningRateScheduler] = {   # pylint: disable=protected-access
        "step": torch.optim.lr_scheduler.StepLR,
        "multi_step": torch.optim.lr_scheduler.MultiStepLR,
        "exponential": torch.optim.lr_scheduler.ExponentialLR,
        "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "cosine": CosineWithRestarts,
        "noam": NoamLR,
}
