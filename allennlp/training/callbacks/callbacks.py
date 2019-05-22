from typing import Set, Dict, List, Union, Iterable
import logging
import math
import traceback
from typing_extensions import Protocol

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import lazy_groups_of
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.data.instance import Instance
from allennlp.models import Model
from allennlp.training import util as training_util
from allennlp.training.callbacks import Callback, Events
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer
from allennlp.training.tensorboard_writer import TensorboardWriter

logger = logging.getLogger(__name__)


class TensorboardHistogramsState(Protocol):
    model: Model
    tensorboard: TensorboardWriter

class LogTensorboardHistograms(Callback[TensorboardHistogramsState]):
    def __init__(self):
        self.histogram_parameters: Set[str] = set()
        self.param_updates: Dict[str, torch.Tensor] = {}

    def __call__(self, event: str, state: TensorboardHistogramsState) -> None:
        if event == Events.TRAINING_START:
            self.histogram_parameters = set(
                    state.model.get_parameters_for_histogram_tensorboard_logging()
            )
        elif event == Events.BATCH_START and state.tensorboard.should_log_histograms_this_batch():
            # get the magnitude of parameter updates for logging
            # We need a copy of current parameters to compute magnitude of updates,
            # and copy them to CPU so large models won't go OOM on the GPU.
            self.param_updates = {name: param.detach().cpu().clone()
                                  for name, param in state.model.named_parameters()}
        elif event == Events.BATCH_END and state.tensorboard.should_log_histograms_this_batch():
            for name, param in state.model.named_parameters():
                self.param_updates[name].sub_(param.detach().cpu())
                update_norm = torch.norm(self.param_updates[name].view(-1, ))
                param_norm = torch.norm(param.view(-1, )).cpu()
                state.tensorboard.add_train_scalar("gradient_update/" + name,
                                                   update_norm / (param_norm + 1e-7))
            self.param_updates.clear()
            state.tensorboard.log_histograms(state.model, self.histogram_parameters)


class TensorboardState(Protocol):
    model: Model
    tensorboard: TensorboardWriter
    optimizer: Optimizer
    batch_grad_norm: float
    batch_group: list
    cumulative_batch_size: int
    train_metrics: dict
    val_metrics: dict


class LogTensorboard(Callback[TensorboardState]):
    def __init__(self, log_batch_size_period: int = None) -> None:
        self.log_batch_size_period = log_batch_size_period
        self.epoch = 1

    def __call__(self, event: str, state: TensorboardState) -> None:
        # At epoch start get parameters for histogram logging
        if event == Events.BATCH_END:
            # Log parameter values to tensorboard
            if state.tensorboard.should_log_this_batch():
                state.tensorboard.log_parameter_and_gradient_statistics(state.model, state.batch_grad_norm)
                state.tensorboard.log_learning_rates(state.model, state.optimizer)

                state.tensorboard.add_train_scalar("loss/loss_train", state.train_metrics["loss"])
                state.tensorboard.log_metrics({"epoch_metrics/" + k: v for k, v in state.train_metrics.items()})


            if self.log_batch_size_period:
                cur_batch = sum([training_util.get_batch_size(batch) for batch in state.batch_group])
                state.cumulative_batch_size += cur_batch
                if (state.batches_this_epoch - 1) % self.log_batch_size_period == 0:
                    average = state.cumulative_batch_size / state.batches_this_epoch
                    logger.info(f"current batch size: {cur_batch} mean batch size: {average}")
                    state.tensorboard.add_train_scalar("current_batch_size", cur_batch)
                    state.tensorboard.add_train_scalar("mean_batch_size", average)

        elif event == Events.EPOCH_END:
            state.tensorboard.log_metrics(state.train_metrics,
                                          val_metrics=state.val_metrics,
                                          log_to_console=True,
                                          epoch=self.epoch)
            self.epoch += 1


class LrsState(Protocol):
    learning_rate_scheduler: LearningRateScheduler
    batch_num_total: int
    epoch_number: int
    val_metrics: dict
    validation_metric: str

class LrsCallback(Callback[LrsState]):
    def __call__(self, event: str, state: LrsState) -> None:
        # Don't do anything if there's no lr_scheduler
        if state.learning_rate_scheduler is None:
            return

        if event == Events.AFTER_BACKWARD:
            state.learning_rate_scheduler.step_batch(state.batch_num_total)
        elif event == Events.EPOCH_END:
            state.learning_rate_scheduler.step(state.val_metrics[state.validation_metric],
                                               state.epoch_number)

class CheckpointState(Protocol):
    checkpointer: Checkpointer
    checkpoint_epoch: Union[int, str]
    model: Model
    metric_tracker: MetricTracker
    epoch_number: int

_DEFAULT_STATE_DICT_ATTRS = ['metric_tracker',
                             'learning_rate_scheduler',
                             'momentum_scheduler',
                             'optimizer']

_DEFAULT_OTHER_ATTRS = ['batch_num_total']


class CheckpointCallback(Callback[CheckpointState]):
    def __init__(self,
                 state_dict_attrs: List[str] = None,
                 other_attrs: List[str] = None) -> None:
        self.state_dict_attrs = state_dict_attrs or _DEFAULT_STATE_DICT_ATTRS
        self.other_attrs = other_attrs or _DEFAULT_OTHER_ATTRS

    def __call__(self, event: str, state: CheckpointState) -> None:
        if event == Events.SAVE_CHECKPOINT:
            training_states = {}
            for attr in self.state_dict_attrs:
                state_attr = getattr(state, attr)
                if state_attr is not None:
                    training_states[attr] = state_attr.state_dict()
            for attr in self.other_attrs:
                training_states[attr] = getattr(state, attr)

            state.checkpointer.save_checkpoint(
                    model_state=state.model.state_dict(),
                    epoch=state.checkpoint_epoch,
                    training_states=training_states,
                    is_best_so_far=state.metric_tracker.is_best_so_far())


        elif event == Events.RESTORE_CHECKPOINT:
            """
            Restores the model and training state from the last saved checkpoint.
            This includes an epoch count and optimizer state, which is serialized separately
            from model parameters. This function should only be used to continue training -
            if you wish to load a model for inference/load parts of a model into a new
            computation graph, you should use the native Pytorch functions:
            `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

            If ``self._serialization_dir`` does not exist or does not contain any checkpointed weights,
            this will do nothing.
            """
            try:
                model_state, training_state = state.checkpointer.restore_checkpoint()
            except RuntimeError:
                traceback.print_exc()
                raise ConfigurationError("Could not recover training from the checkpoint.  "
                                         "Did you mean to output to a different serialization directory "
                                         "or delete the existing serialization directory?")

            if not training_state:
                # No checkpoint to restore, start at 0
                state.epoch_number = 0
                return

            state.model.load_state_dict(model_state)

            for attr in self.state_dict_attrs:
                state_attr = getattr(state, attr)
                if state_attr is not None:
                    state_attr.load_state_dict(training_state[attr])

            for attr in self.other_attrs:
                setattr(state, attr, training_state[attr])

            if isinstance(training_state["epoch"], int):
                state.epoch_number = training_state["epoch"] + 1
            else:
                state.epoch_number = int(training_state["epoch"].split('.')[0]) + 1


class MovingAverageState(Protocol):
    moving_average: MovingAverage
    batch_num_total: int

class MovingAverageCallback(Callback[MovingAverageState]):
    def __call__(self, event: str, state: MovingAverageState) -> None:
        if state.moving_average is None:
            return

        if event == Events.BATCH_END:
            state.moving_average.apply(state.batch_num_total)

        elif event in [Events.BEFORE_SAVE_CHECKPOINT, Events.BEFORE_VALIDATE]:
            # If moving averages are used for parameters, we save
            # the moving average values into checkpoint, instead of the current values.
            state.moving_average.assign_average_value()

        elif event in [Events.AFTER_SAVE_CHECKPOINT, Events.AFTER_VALIDATE]:
            # Restore the original values for parameters so that training will not be affected.
            state.moving_average.restore()


class ValidateState(Protocol):
    model: Model
    validation_iterator: DataIterator
    validation_data: Iterable[Instance]
    validation_metric: str
    metric_tracker: MetricTracker
    val_metrics: dict
    _cuda_devices: list

    # pylint: disable=unused-argument,no-self-use,multiple-statements,pointless-statement
    def batch_loss(self, batch_group: List[TensorDict], for_training: bool) -> torch.Tensor: ...

class Validate(Callback[ValidateState]):
    def __call__(self, event: str, state: ValidateState) -> None:
        if event == Events.VALIDATE:
            if state.validation_data is None:
                return

            with torch.no_grad():
                # We have a validation set, so compute all the metrics on it.
                logger.info("Validating")

                state.model.eval()

                num_gpus = len(state._cuda_devices)  # pylint: disable=protected-access

                raw_val_generator = state.validation_iterator(
                        state.validation_data,
                        num_epochs=1,
                        shuffle=False)
                val_generator = lazy_groups_of(raw_val_generator, num_gpus)
                num_validation_batches = math.ceil(
                        state.validation_iterator.get_num_batches(state.validation_data) / num_gpus)
                val_generator_tqdm = Tqdm.tqdm(val_generator,
                                               total=num_validation_batches)

                batches_this_epoch = 0
                val_loss = 0
                for batch_group in val_generator_tqdm:

                    loss = state.batch_loss(batch_group, for_training=False)
                    if loss is not None:
                        # You shouldn't necessarily have to compute a loss for validation, so we allow for
                        # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                        # currently only used as the divisor for the loss function, so we can safely only
                        # count those batches for which we actually have a loss.  If this variable ever
                        # gets used for something else, we might need to change things around a bit.
                        batches_this_epoch += 1
                        val_loss += loss.detach().cpu().numpy()

                # Update the description with the latest metrics
                val_metrics = training_util.get_metrics(state.model, val_loss, batches_this_epoch)
                description = training_util.description_from_metrics(val_metrics)
                val_generator_tqdm.set_description(description, refresh=False)

                state.val_metrics = training_util.get_metrics(state.model,
                                                              val_loss,
                                                              batches_this_epoch,
                                                              reset=True)

                # Check validation metric for early stopping
                this_epoch_val_metric = state.val_metrics[state.validation_metric]
                state.metric_tracker.add_metric(this_epoch_val_metric)
