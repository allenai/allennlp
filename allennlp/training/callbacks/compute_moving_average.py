# pylint: disable=unused-variable,arguments-differ,unused-argument
from typing import TYPE_CHECKING

from allennlp.common.params import Params
from allennlp.models import Model
from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.training.callbacks.events import Events
from allennlp.training.moving_average import MovingAverage

if TYPE_CHECKING:
    from allennlp.training.callback_trainer import CallbackTrainer  # pylint:disable=unused-import


@Callback.register("compute_moving_average")
class ComputeMovingAverage(Callback):
    """
    Callback that handles computing a moving average the model parameters.
    The timings for this one are slightly delicate, as every time we either
    checkpoint or validate we want to first load the values from the moving
    average, and afterward we want to restore the values.

    Parameters
    ----------
    moving_average : ``MovingAverage``
        The moving average tracker to manage
    """
    # pylint: disable=no-self-use
    def __init__(self, moving_average: MovingAverage) -> None:
        self.moving_average = moving_average

    @handle_event(Events.BATCH_END)
    def apply_moving_average(self, trainer: 'CallbackTrainer'):
        self.moving_average.apply(trainer.batch_num_total)

    @handle_event(Events.SAVE_CHECKPOINT, priority=-1000)
    def assign_average_value_before_saving(self, trainer: 'CallbackTrainer'):
        # If moving averages are used for parameters, we save
        # the moving average values into checkpoint, instead of the current values.
        self.moving_average.assign_average_value()

    @handle_event(Events.VALIDATE, priority=-1000)
    def assign_average_value_before_validation(self, trainer: 'CallbackTrainer'):
        # If moving averages are used for parameters, we save
        # the moving average values into checkpoint, instead of the current values.
        self.moving_average.assign_average_value()

    @handle_event(Events.SAVE_CHECKPOINT, priority=1000)
    def restore_values_after_saving(self, trainer: 'CallbackTrainer'):
        # Restore the original values for parameters so that training will not be affected.
        self.moving_average.restore()

    @handle_event(Events.VALIDATE, priority=1000)
    def restore_values_after_validation(self, _trainer: 'CallbackTrainer'):
        # Restore the original values for parameters so that training will not be affected.
        self.moving_average.restore()

    @classmethod
    def from_params(cls, params: Params, model: Model) -> 'MovingAverageCallback':  # type: ignore
        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        moving_average = MovingAverage.from_params(params.pop("moving_average"), parameters=parameters)
        return ComputeMovingAverage(moving_average)
