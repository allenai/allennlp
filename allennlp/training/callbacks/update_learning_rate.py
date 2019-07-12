from typing import TYPE_CHECKING

import torch

from allennlp.common.params import Params
from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.training.callbacks.events import Events
from allennlp.training.learning_rate_schedulers import LearningRateScheduler

if TYPE_CHECKING:
    from allennlp.training.callback_trainer import CallbackTrainer  # pylint:disable=unused-import


@Callback.register("update_learning_rate")
class UpdateLearningRate(Callback):
    """
    Callback that runs the learning rate scheduler.

    Parameters
    ----------
    learning_rate_scheduler : ``LearningRateScheduler``
        The scheduler to handler.
    """
    def __init__(self, learning_rate_scheduler: LearningRateScheduler) -> None:
        self.learning_rate_scheduler = learning_rate_scheduler

    @handle_event(Events.BACKWARD, priority=1000)
    def step_batch(self, trainer: 'CallbackTrainer'):
        self.learning_rate_scheduler.step_batch(trainer.batch_num_total)

    @handle_event(Events.EPOCH_END)
    def step(self, trainer: 'CallbackTrainer'):
        self.learning_rate_scheduler.step(trainer.latest_val_metric, trainer.epoch_number)

    def get_training_state(self) -> dict:
        """
        We need to persist the learning_rate_scheduler state as training state.
        """
        return {"learning_rate_scheduler": self.learning_rate_scheduler.state_dict()}

    def restore_training_state(self, training_state: dict) -> None:
        state_dict = training_state.pop("learning_rate_scheduler", None)

        if state_dict:
            self.learning_rate_scheduler.load_state_dict(state_dict)

    @classmethod
    def from_params(cls,                # type: ignore
                    params: Params,
                    optimizer: torch.optim.Optimizer) -> 'UpdateLearningRate':
        # pylint: disable=arguments-differ
        return cls(LearningRateScheduler.from_params(params=params.pop("learning_rate_scheduler"),
                                                     optimizer=optimizer))
