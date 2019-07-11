from typing import TYPE_CHECKING
import logging

from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.training.callbacks.events import Events

if TYPE_CHECKING:
    from allennlp.training.callback_trainer import CallbackTrainer  # pylint:disable=unused-import

logger = logging.getLogger(__name__)


@Callback.register("gradient_accumulation")
class GradientAccumulation(Callback):
    """
    Applies gradient accumulation by manually setting
    trainer.is_start_of_batch and trainer.is_end_of_batch.

    Parameters
    ----------
    gradient_accumulation_period : ``int``
        How many forward / backward passes make up a single "batch".
    """
    def __init__(self,
                 gradient_accumulation_period: int) -> None:
        self.gradient_accumulation_period = gradient_accumulation_period
        self.count = 0

    @handle_event(Events.TRAINING_START)
    def set_loss_scale(self, trainer: 'CallbackTrainer') -> None:
        # If the gradient accumulation period is (say) 3,
        # we are accumulating the losses from 3 batch_groups,
        # so we want to divide them by 3.
        trainer.loss_scale /= self.gradient_accumulation_period

    @handle_event(Events.EPOCH_START)
    def reset_counter(self, trainer: 'CallbackTrainer') -> None:
        self.count = 0

        # It's definitely the start of a (synthetic) batch, but it's only the end
        # if the gradient accumulation period equals 1.
        trainer.is_start_of_batch = True
        trainer.is_end_of_batch = self.gradient_accumulation_period == 1

    @handle_event(Events.FORWARD)
    def increment_counter(self, trainer: 'CallbackTrainer') -> None:
        self.count += 1

        if self.count >= self.gradient_accumulation_period:
            # This is e.g. batch_group 3 / 3.
            # We reset the count,
            # set is_start_of_batch = True (because it won't be checked until next batch_group)
            # and set is_end_of_batch = True (because it will be checked at the end of this batch_group)
            self.count = 0
            trainer.is_start_of_batch = trainer.is_end_of_batch = True
        else:
            # This is e.g. batch_group 1 / 3 or 2 / 3.
            # so both flags should be set to false.
            # (In the 1/3 case we've already done the "start of batch" check.)
            trainer.is_start_of_batch = trainer.is_end_of_batch = False
