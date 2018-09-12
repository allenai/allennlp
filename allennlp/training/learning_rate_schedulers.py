"""
AllenNLP uses most
`PyTorch learning rate schedulers <http://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate>`_,
with a thin wrapper to allow registering them and instantiating them ``from_params``.

The available learning rate schedulers from PyTorch are

* `"step" <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.StepLR>`_
* `"multi_step" <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.MultiStepLR>`_
* `"exponential" <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.ExponentialLR>`_
* `"reduce_on_plateau" <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau>`_

In addition, AllenNLP also provides a Noam schedule and `cosine with restarts
<https://arxiv.org/abs/1608.03983>`_, which are registered as "noam" and "cosine", respectively.
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


class SlantedTriangular(torch.optim.lr_scheduler._LRScheduler): # pylint: disable=protected-access
    """
    Implements the Slanted Triangular Learning Rate schedule with optional gradual
    unfreezing. The schedule corresponds to first linearly increasing the learning
    rate and annealing the learning based on a fixed ratio.

    If we gradually unfreeze, then in the first epoch of training, only the top
    layer is trained; in the second epoch, the top two layers are trained, etc.
    During freezing, the learning rate is increased and annealed over one epoch.
    After freezing finished, the learning rate is increased and annealed over
    the remaining training iterations.

    Note that with this schedule, early stopping should typically be avoided.

    Parameters
    ----------
    num_epochs : ``int``, required.
        The total number of epochs for which the model should be trained.
    num_steps_per_epoch: ``int``, required.
        The number of steps (updates, batches) per training epoch.
    cut_frac: ``float``, optional (default = 0.1).
        The fraction of the steps to increase the learning rate.
    ratio: ``float``, optional (default = 32).
        The ratio of the smallest to the (largest) base learning rate.
    gradual_unfreezing: ``bool``, optional (default = False).
        Whether gradual unfreezing should be used.
    discriminative_fine_tuning: ``bool``, optional (default = False).
        Whether discriminative fine-tuning (different learning rates per layer)
        are used.
    decay_factor: ``float``, optional (default = 0.38).
        The decay factor by which the learning rate is reduced with
        discriminative fine-tuning when going a layer deeper.
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 num_epochs: int,
                 num_steps_per_epoch: int,
                 cut_frac: float = 0.1,
                 ratio: int = 32,
                 last_epoch: int = -1,
                 gradual_unfreezing: bool = False,
                 discriminative_fine_tuning: bool = False,
                 decay_factor: float = 0.38) -> None:
        self.num_epochs = num_epochs
        self.num_steps_per_epoch = num_steps_per_epoch
        self.cut_frac = cut_frac
        self.ratio = ratio
        self.gradual_unfreezing = gradual_unfreezing
        self.freezing_current = self.gradual_unfreezing
        self.is_first_epoch = True
        if self.gradual_unfreezing:
            assert not optimizer.param_groups[-1]["params"], \
                "The default group should be empty."
        if self.gradual_unfreezing or discriminative_fine_tuning:
            assert len(optimizer.param_groups) > 2, \
                "There should be at least 3 param_groups (2 + empty default group)" \
                " for gradual unfreezing / discriminative fine-tuning to make sense."
        super().__init__(optimizer, last_epoch=last_epoch)
        if discriminative_fine_tuning:
            # skip the last param_group if it is has no parameters
            exponent = 0
            for i in range(len(self.base_lrs)-1, -1, -1):
                param_group = optimizer.param_groups[i]
                if param_group['params']:
                    param_group['lr'] = self.base_lrs[i] * decay_factor ** exponent
                    self.base_lrs[i] = param_group['lr']
                    exponent += 1
        # set up for the first batch
        self.step_batch(0)

    def step(self, epoch=None):
        if self.gradual_unfreezing:
            # the method is called once when initialising before the
            # first epoch (epoch 0) and then always at the end of each
            # epoch; so the first time, with epoch id 0, we want to set
            # up for epoch #1; the second time, still with epoch id 0,
            # we want to set up for epoch #2, etc.
            num_layers_to_unfreeze = epoch + 1 if self.is_first_epoch else epoch + 2
            if self.is_first_epoch:
                self.is_first_epoch = False
            if num_layers_to_unfreeze >= len(self.optimizer.param_groups)-1:
                logger.info('Gradual unfreezing finished. Training all layers.')
                self.freezing_current = False
            else:
                logger.info(f'Gradual unfreezing. Training only the top {num_layers_to_unfreeze} layers.')
            for i, param_group in enumerate(reversed(self.optimizer.param_groups)):
                for param in param_group["params"]:
                    # i = 0 is the default group; we care about i > 0
                    param.requires_grad = bool(i <= num_layers_to_unfreeze)

    def step_batch(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, learning_rate in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = learning_rate

    def get_lr(self):
        if self.freezing_current:
            # if we still freeze, we restrict the schedule to the current epoch
            num_steps = self.num_steps_per_epoch
            step = self.last_epoch % num_steps
        else:
            # otherwise we use the schedule for the rest of training
            if not self.gradual_unfreezing:
                frozen_steps = 0
            else:
                frozen_steps = (len(self.optimizer.param_groups) - 2) * self.num_steps_per_epoch
            num_steps = self.num_epochs * self.num_steps_per_epoch - frozen_steps
            step = (self.last_epoch - frozen_steps) % num_steps
        cut = int(num_steps * self.cut_frac)
        prop = step / cut if step < cut else 1 - (step - cut) / (num_steps - cut)
        return [lr * (1 + prop * (self.ratio - 1)) / self.ratio for lr in self.base_lrs]


class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):  # pylint: disable=protected-access
    """
    Cosine annealing with restarts.

    This is decribed in the paper https://arxiv.org/abs/1608.03983.

    Parameters
    ----------
    optimizer : ``torch.optim.Optimizer``

    t_initial : ``int``
        The number of iterations within the first cycle.

    t_mul : ``float``, optional (default=1)
        Determines the number of iterations in the i-th decay cycle, which is the
        length of the last cycle multiplied by ``t_mul``.

    eta_min : ``float``, optional (default=0)
        The minimum learning rate.

    eta_mul : ``float``, optional (default=1)
        Determines the initial learning rate for the i-th decay cycle, which is the
        last initial learning rate multiplied by ``m_mul``.

    last_epoch : ``int``, optional (default=-1)
        The index of the last epoch. This is used when restarting.

    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 t_mul: float = 1.,
                 eta_min: float = 0.,
                 eta_mul: float = 1.,
                 last_epoch: int = -1) -> None:
        assert t_initial > 0
        assert eta_min >= 0
        if t_initial == 1 and t_mul == 1 and eta_mul == 1:
            logger.warning("Cosine annealing scheduler will have no effect on the learning "
                           "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial
        self.t_mul = t_mul
        self.eta_min = eta_min
        self.eta_mul = eta_mul
        self._last_restart: int = 0
        self._cycle_counter: int = 0
        self._cycle_len: int = t_initial
        self._n_restarts: int = 0
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

        if self._cycle_counter % self._cycle_len == 0:
            self._n_restarts += 1
            self._cycle_counter = 0
            self._last_restart = step

        base_lrs = [lr * self.eta_mul**self._n_restarts for lr in self.base_lrs]
        self._cycle_len = int(self.t_initial * self.t_mul**self._n_restarts)

        lrs = [
                self.eta_min + ((lr - self.eta_min) / 2) * (
                        np.cos(np.pi * (self._cycle_counter % self._cycle_len) / self._cycle_len) + 1
                )
                for lr in base_lrs
        ]

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
        "slanted_triangular": SlantedTriangular,
}
