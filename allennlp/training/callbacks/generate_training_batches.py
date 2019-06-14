# pylint: disable=unused-variable,arguments-differ
from typing import Iterable, TYPE_CHECKING
import logging
import math

from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators import DataIterator
from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.training.callbacks.events import Events

if TYPE_CHECKING:
    from allennlp.training.callback_trainer import CallbackTrainer  # pylint:disable=unused-import

logger = logging.getLogger(__name__)


@Callback.register("generate_training_batches")
class GenerateTrainingBatches(Callback):
    """
    Generates the training batches for one epoch.

    Parameters
    ----------
    instances : Iterable[Instance]
        The instances in the train dataset.
    iterator : DataIterator
        The iterator to use for batching.
    shuffle : bool, optional (default = True)
        Whether to shuffle the instances each epoch.
    """
    def __init__(self,
                 instances: Iterable[Instance],
                 iterator: DataIterator,
                 shuffle: bool = True) -> None:
        self.instances = instances
        self.iterator = iterator
        self.shuffle = shuffle

    @handle_event(Events.EPOCH_START)
    def generate_batches(self, trainer: 'CallbackTrainer'):
        # pylint: disable=protected-access
        num_gpus = len(trainer._cuda_devices)

        raw_train_generator = self.iterator(self.instances,
                                            num_epochs=1,
                                            shuffle=self.shuffle)
        trainer.training_batches = lazy_groups_of(raw_train_generator, num_gpus)
        trainer.num_training_batches = math.ceil(self.iterator.get_num_batches(self.instances) / num_gpus)
