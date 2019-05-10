import math
import os
from typing import Any, Dict, List, Union

from allennlp.common import Params
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import lazy_groups_of
from allennlp.models import Model
from allennlp.training import util as training_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.trainer import TrainerPieces
from allennlp.training.trainer_base import TrainerBase


@TrainerBase.register("no_op")
class NoOpTrainer(TrainerBase):
    def __init__(self, serialization_dir: str, model: Model, cuda_device: Union[int, List] = -1) -> None:
        """
        A trivial trainer to assist in making model archives for models that do not actually
        require training. For instance, a majority class baseline.
        """

        super().__init__(serialization_dir, cuda_device=cuda_device)
        self.model = model
        # TODO: add iterator and so on.

    @classmethod
    def from_params(cls,  # type: ignore
                    params: Params,
                    serialization_dir: str,
                    recover: bool = False):
        # pylint: disable=arguments-differ
        pieces = TrainerPieces.from_params(params, serialization_dir, recover)  # pylint: disable=no-member
        return NoOpTrainer(serialization_dir, pieces.model)

    def _run_on_validation(self) -> Dict[str, float]:
        val_metrics = {}

        if self._validation_iterator is not None:
            val_iterator = self._validation_iterator
        else:
            val_iterator = self.iterator

        num_gpus = len(self._cuda_devices)

        raw_val_generator = val_iterator(self._validation_data,
                                         num_epochs=1,
                                         shuffle=False)
        val_generator = lazy_groups_of(raw_val_generator, num_gpus)
        num_validation_batches = math.ceil(val_iterator.get_num_batches(self._validation_data) / num_gpus)
        val_generator_tqdm = Tqdm.tqdm(val_generator,
                                       total=num_validation_batches)
        batches_this_epoch = 0
        val_loss = 0
        for batch_group in val_generator_tqdm:

            loss = self.batch_loss(batch_group, for_training=False)
            if loss is not None:
                # You shouldn't necessarily have to compute a loss for validation, so we allow for
                # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                # currently only used as the divisor for the loss function, so we can safely only
                # count those batches for which we actually have a loss.  If this variable ever
                # gets used for something else, we might need to change things around a bit.
                batches_this_epoch += 1
                val_loss += loss.detach().cpu().numpy()

            # Update the description with the latest metrics
            val_metrics = self.model.get_metrics(reset=True)
            description = training_util.description_from_metrics(val_metrics)
            val_generator_tqdm.set_description(description, refresh=False)

        return val_metrics

    def train(self) -> Dict[str, Any]:
        self.model.vocab.save_to_files(os.path.join(self._serialization_dir, "vocabulary"))

        checkpointer = Checkpointer(self._serialization_dir)
        checkpointer.save_checkpoint(epoch=0,
                                     model_state=self.model.state_dict(),
                                     training_states={},
                                     is_best_so_far=True)

        # TODO: pass the model through the training data
        training_util.evaluate(self.model, instances, self.iterator, cuda_device=self._cuda_devices,
                               batch_weight_key='')

        training_metrics = self.model.get_metrics(reset=True)
        metrics = {f"training_{key}": value for key, value in training_metrics.items()}

        if self._validation_data:
            val_metrics = self._run_on_validation()
            for key, value in val_metrics.items():
                metrics[f"validation_{key}"] = value
                metrics[f"best_validation_{key}"] = value

        return metrics
