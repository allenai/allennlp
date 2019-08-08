from typing import TYPE_CHECKING

from allennlp.common.params import Params
from allennlp.models.model import Model
from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.training.callbacks.events import Events
from allennlp.training.moving_average import MovingAverage

if TYPE_CHECKING:
    from allennlp.training.callback_trainer import CallbackTrainer  # pylint:disable=unused-import


@Callback.register("update_moving_average")
class UpdateMovingAverage(Callback):
    """
    Callback that orchestrates checkpointing of your model and training state.

    Parameters
    ----------
    moving_aveage : ``MovingAverage``
        The MovingAverage object to update.
    """
    def __init__(self, moving_average: MovingAverage) -> None:
        self.moving_average = moving_average

    @handle_event(Events.BATCH_END, priority=-1000)
    def apply_moving_average(self, trainer: 'CallbackTrainer') -> None:
        self.moving_average.apply(trainer.batch_num_total)

    @classmethod
    def from_params(cls, params: Params, model: Model) -> 'UpdateMovingAverage':  # type: ignore
        # pylint: disable=arguments-differ
        moving_average_params = params.pop("moving_average")
        model_parameters = [[name, param] for name, param in model.named_parameters() if param.requires_grad]
        moving_average = MovingAverage.from_params(params=moving_average_params, parameters=model_parameters)

        return UpdateMovingAverage(moving_average)
