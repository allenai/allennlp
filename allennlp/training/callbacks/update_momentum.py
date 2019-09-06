from typing import TYPE_CHECKING

import torch

from allennlp.common.params import Params
from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.training.callbacks.events import Events
from allennlp.training.momentum_schedulers import MomentumScheduler

if TYPE_CHECKING:
    from allennlp.training.callback_trainer import CallbackTrainer  # pylint:disable=unused-import


@Callback.register("update_momentum")
class UpdateMomentum(Callback):
    """
    Callback that runs a Momentum Scheduler.

    Parameters
    ----------
    momentum_scheduler : ``MomentumScheduler``
        The momentum scheduler to run.
    """
    def __init__(self, momentum_scheduler: MomentumScheduler) -> None:
        self.momentum_scheduler = momentum_scheduler

    @handle_event(Events.BACKWARD, priority=1000)
    def step_batch(self, trainer: 'CallbackTrainer'):
        self.momentum_scheduler.step_batch(trainer.batch_num_total)

    @handle_event(Events.EPOCH_END)
    def step(self, trainer: 'CallbackTrainer'):
        self.momentum_scheduler.step(trainer.latest_val_metric, trainer.epoch_number)

    def get_training_state(self) -> dict:
        return {"momentum_scheduler": self.momentum_scheduler.state_dict()}

    def restore_training_state(self, training_state: dict) -> None:
        state_dict = training_state.pop("momentum_scheduler", None)

        if state_dict:
            self.momentum_scheduler.load_state_dict(state_dict)

    @classmethod
    def from_params(cls, params: Params, optimizer: torch.optim.Optimizer) -> 'UpdateMomentum':  # type: ignore
        # pylint: disable=arguments-differ
        return cls(MomentumScheduler.from_params(params=params.pop("momentum_scheduler"),
                                                 optimizer=optimizer))
