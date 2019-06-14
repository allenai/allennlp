from typing import Iterable, TYPE_CHECKING
import logging
import math

import torch

from allennlp.common.tqdm import Tqdm
from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators import DataIterator
from allennlp.training import util as training_util
from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.training.callbacks.events import Events

if TYPE_CHECKING:
    from allennlp.training.callback_trainer import CallbackTrainer  # pylint:disable=unused-import

logger = logging.getLogger(__name__)


@Callback.register("validate")
class Validate(Callback):
    """
    Evaluates the trainer's ``Model`` using the provided validation dataset.
    Uses the results to populate trainer.val_metrics.

    Parameters
    ----------
    validation_data : ``Iterable[Instance]``
        The instances in the validation dataset.
    validation_iterator : ``DataIterator``
        The iterator to use in the evaluation.
    """
    def __init__(self,
                 validation_data: Iterable[Instance],
                 validation_iterator: DataIterator) -> None:
        self.instances = validation_data
        self.iterator = validation_iterator

    @handle_event(Events.TRAINING_START)
    def set_validate(self, trainer: 'CallbackTrainer'):
        # pylint: disable=no-self-use
        trainer.validate = True

    @handle_event(Events.VALIDATE)
    def validate(self, trainer: 'CallbackTrainer'):
        with torch.no_grad():
            # We have a validation set, so compute all the metrics on it.
            logger.info("Validating")

            trainer.model.eval()

            num_gpus = len(trainer._cuda_devices)  # pylint: disable=protected-access

            raw_val_generator = self.iterator(self.instances,
                                              num_epochs=1,
                                              shuffle=False)
            val_generator = lazy_groups_of(raw_val_generator, num_gpus)
            num_validation_batches = math.ceil(
                    self.iterator.get_num_batches(self.instances) / num_gpus)
            val_generator_tqdm = Tqdm.tqdm(val_generator,
                                           total=num_validation_batches)

            batches_this_epoch = 0
            val_loss = 0
            for batch_group in val_generator_tqdm:

                loss = trainer.batch_loss(batch_group, for_training=False)
                if loss is not None:
                    # You shouldn't necessarily have to compute a loss for validation, so we allow for
                    # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                    # currently only used as the divisor for the loss function, so we can safely only
                    # count those batches for which we actually have a loss.  If this variable ever
                    # gets used for something else, we might need to change things around a bit.
                    batches_this_epoch += 1
                    val_loss += loss.detach().cpu().numpy()

                # Update the description with the latest metrics
                val_metrics = training_util.get_metrics(trainer.model, val_loss, batches_this_epoch)
                description = training_util.description_from_metrics(val_metrics)
                val_generator_tqdm.set_description(description, refresh=False)

            trainer.val_metrics = training_util.get_metrics(trainer.model,
                                                            val_loss,
                                                            batches_this_epoch,
                                                            reset=True)
