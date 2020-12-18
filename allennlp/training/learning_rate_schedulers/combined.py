from typing import Dict, Any, List, Tuple, Optional

from overrides import overrides
import torch

from allennlp.common.lazy import Lazy
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler


@LearningRateScheduler.register("combined")
class CombinedLearningRateScheduler(LearningRateScheduler):
    """
    This `LearningRateScheduler` can be used to apply an arbitrary number of other schedulers
    one after the other.

    These schedulers are defined though the `schedulers` parameter, which takes a list
    of `Tuple[int, Lazy[LearningRateScheduler]]`. The first field of the tuple, the `int`,
    specifies how many epochs the corresponding scheduler will be used before the next
    scheduler takes its place.

    While it usually makes sense for the sum

    ```python
    sum(n_epochs for (n_epochs, _) in schedulers)
    ```

    to equal the total number of training epochs, it is not a requirement.
    If training continues beyond the last defined scheduler, both `step()` and `step_batch()`
    will be a no-op.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        schedulers: List[Tuple[int, Lazy[LearningRateScheduler]]],
        num_steps_per_epoch: Optional[int] = None,
        last_epoch: int = -1,
    ) -> None:
        super().__init__(optimizer, last_epoch=last_epoch)
        self.num_steps_per_epoch = num_steps_per_epoch
        self.schedulers = schedulers
        # This is used to know when we need to update `self._current_scheduler`
        # by comparing it to `self.last_epoch`, and so to start with it needs to
        # not equal `self.last_epoch`.
        self._last_epoch_updated = -2
        self._current_scheduler: Optional[LearningRateScheduler] = None
        self._current_scheduler_first_epoch: Optional[int] = None
        # We call this here in order to initialize the current scheduler now, since some schedulers
        # modify the LR when they are initialized.
        self.current_scheduler

    @property
    def current_scheduler(self) -> Optional[LearningRateScheduler]:
        if self._last_epoch_updated != self.last_epoch:
            current_epoch = self.last_epoch + 1
            scheduler_first_epoch, scheduler_last_epoch = 0, -1
            for scheduler_epochs, lazy_scheduler in self.schedulers:
                scheduler_last_epoch += scheduler_epochs

                # Is it time for a new scheduler?
                if current_epoch == scheduler_first_epoch or (
                    self._current_scheduler_first_epoch != scheduler_first_epoch
                    and scheduler_first_epoch <= current_epoch <= scheduler_last_epoch
                ):
                    # Reset the base values of the LR to whatever they're currently at.
                    for group in self.optimizer.param_groups:
                        group[self._initial_param_group_field] = group[self.param_group_field]
                    self._current_scheduler = lazy_scheduler.construct(
                        optimizer=self.optimizer,
                        num_epochs=scheduler_epochs,
                        num_steps_per_epoch=self.num_steps_per_epoch,
                    )
                    self._current_scheduler_first_epoch = scheduler_first_epoch
                    break

                scheduler_first_epoch = scheduler_last_epoch + 1
            else:
                # If we didn't break out of the loop, then we might have trained past
                # the last defined scheduler, so we're not going to use a scheduler anymore.
                if current_epoch > scheduler_last_epoch:
                    self._current_scheduler = None
        self._last_epoch_updated = self.last_epoch
        return self._current_scheduler

    @overrides
    def state_dict(self) -> Dict[str, Any]:
        current_scheduler = self.current_scheduler
        return {
            "last_epoch": self.last_epoch,
            "num_steps_per_epoch": self.num_steps_per_epoch,
            "current_scheduler": None
            if current_scheduler is None
            else current_scheduler.state_dict(),
        }

    @overrides
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.last_epoch = state_dict["last_epoch"]
        self.num_steps_per_epoch = state_dict["num_steps_per_epoch"]
        if self.current_scheduler is not None:
            assert state_dict["current_scheduler"] is not None
            self.current_scheduler.load_state_dict(state_dict["current_scheduler"])

    @overrides
    def get_values(self):
        """
        This should never be called directly.
        """
        raise NotImplementedError

    @overrides
    def step_batch(self, batch_num_total: int = None) -> None:
        if self.current_scheduler is not None:
            self.current_scheduler.step_batch(batch_num_total)

    @overrides
    def step(self, metric: float = None) -> None:
        self.last_epoch += 1
        self.metric = metric
        if self.current_scheduler is not None:
            self.current_scheduler.step(metric)
