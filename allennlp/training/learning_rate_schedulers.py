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

from typing import Optional

import torch.optim.lr_scheduler
from overrides import overrides
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.registrable import Registrable


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


class STLR(torch.optim.lr_scheduler._LRScheduler): # pylint: disable=protected-access
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
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 num_epochs: int,
                 num_steps_per_epoch: int,
                 cut_frac: float = 0.1,
                 ratio: int = 32,
                 lr: float = 0.01,
                 last_epoch: int = -1,
                 gradual_unfreezing: bool = False) -> None:
        self.num_epochs = num_epochs
        self.num_steps_per_epoch = num_steps_per_epoch
        self.cut_frac = cut_frac
        self.ratio = ratio
        self.lr = lr
        self.gradual_unfreezing = gradual_unfreezing
        self.freezing_current = self.gradual_unfreezing
        self.first_epoch = True
        super().__init__(optimizer, last_epoch=last_epoch)

    def step(self, epoch=None):
        if self.gradual_unfreezing:
            assert not self.optimizer.param_groups[-1]["params"], \
                "The default group should be empty."
            # the method is called once when initialising before the
            # first epoch (epoch 0) and then always at the end of each
            # epoch; so the first time, with epoch id 0, we want to set
            # up for epoch #1; the second time, still with epoch id 0,
            # we want to set up for epoch #2, etc.
            epoch_no = epoch+1 if self.first_epoch else epoch+2
            if self.first_epoch:
                self.first_epoch = False
            if epoch_no >= len(self.optimizer.param_groups)-1:
                print('Gradual unfreezing finished. Training all layers.')
                self.freezing_current = False
            else:
                print(f'Gradual unfreezing. Training only the top {epoch_no} layers.')
            for i, param_group in enumerate(reversed(self.optimizer.param_groups)):
                for param in param_group["params"]:
                    # i = 0 is the default group; we care about i > 0
                    if i <= epoch_no:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

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
            step = max(self.last_epoch, 1) % self.num_steps_per_epoch
        else:
            # otherwise we use the schedule for the rest of training
            steps_while_frozen = (len(self.optimizer.param_groups)-1) * self.num_steps_per_epoch
            num_steps = self.num_epochs * self.num_steps_per_epoch - steps_while_frozen
            step = max(self.last_epoch, 1) - steps_while_frozen
        cut = int(num_steps * self.cut_frac)
        if cut == step:
            print()
        p = step / cut if step < cut else 1 - (step - cut) / (num_steps - cut)
        lrs = [lr * (1 + p * (self.ratio - 1)) / self.ratio for lr in self.base_lrs]
        return lrs

# We just use the Pytorch LRSchedulers, so here we force them into
# Registry._registry so we can build them from params.
Registrable._registry[LearningRateScheduler] = {   # pylint: disable=protected-access
        "step": torch.optim.lr_scheduler.StepLR,
        "multi_step": torch.optim.lr_scheduler.MultiStepLR,
        "exponential": torch.optim.lr_scheduler.ExponentialLR,
        "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
        "noam": NoamLR,
        "stlr": STLR
}
