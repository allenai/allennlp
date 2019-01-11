import logging

import numpy as np
import torch

from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler


logger = logging.getLogger(__name__)


@LearningRateScheduler.register("cosine")
class CosineWithRestarts(LearningRateScheduler):
    """
    Cosine annealing with restarts.

    This is described in the paper https://arxiv.org/abs/1608.03983.

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
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time ``self.get_lr()`` was called,
        # since ``torch.optim.lr_scheduler._LRScheduler`` will call ``self.get_lr()``
        # when first initialized, but the learning rate should remain unchanged
        # for the first epoch.
        if not self._initialized:
            self._initialized = True
            return self.base_values

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        if self._cycle_counter % self._cycle_len == 0:
            self._n_restarts += 1
            self._cycle_counter = 0
            self._last_restart = step

        base_lrs = [lr * self.eta_mul**self._n_restarts for lr in self.base_values]
        self._cycle_len = int(self.t_initial * self.t_mul**self._n_restarts)

        lrs = [
                self.eta_min + ((lr - self.eta_min) / 2) * (
                        np.cos(np.pi * (self._cycle_counter % self._cycle_len) / self._cycle_len) + 1
                )
                for lr in base_lrs
        ]

        return lrs

    def step(self, metric: float = None, epoch: int = None) -> None:
        if epoch is None:
            self.last_epoch += 1  # type: ignore
        else:
            self.last_epoch = epoch
        for param_group, learning_rate in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = learning_rate
