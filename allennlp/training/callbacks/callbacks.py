from typing import Set, Dict, List, Tuple
import copy
import datetime
import logging
import math
import os
import time
import traceback

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import lazy_groups_of, dump_metrics, gpu_memory_mb, peak_memory_mb
from allennlp.models import Model
from allennlp.training.optimizers import Optimizer
from allennlp.training import util as training_util
from allennlp.training import trainer2  # pylint: disable=unused-import
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.callbacks import Callback, Events
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.tensorboard_writer import TensorboardWriter

logger = logging.getLogger(__name__)


@Callback.register("log_tensorboard")
class LogTensorboard(Callback['trainer2.Trainer']):
    def __init__(self,
                 tensorboard: TensorboardWriter,
                 log_batch_size_period: int = None) -> None:
        self.log_batch_size_period = log_batch_size_period
        self.tensorboard = tensorboard

        self.epoch = 1
        self.cumulative_batch_size = 0

        # For logging histograms
        self.histogram_parameters: Set[str] = set()
        self.param_updates: Dict[str, torch.Tensor] = {}

    def __call__(self, event: str, state: 'trainer2.Trainer') -> None:
        if event == Events.TRAINING_START:
            # Bad hack to get the tensorboard instance to know about the trainer
            # pylint: disable=protected-access
            self.tensorboard._get_batch_num_total = lambda: state.batch_num_total

            # Get histogram parameters
            self.histogram_parameters = set(
                    state.model.get_parameters_for_histogram_tensorboard_logging()
            )

            # Enable activation logging.
            if self.tensorboard._histogram_interval is not None:
                self.tensorboard.enable_activation_logging(state.model)

        elif event == Events.BATCH_START and self.tensorboard.should_log_histograms_this_batch():
            # get the magnitude of parameter updates for logging
            # We need a copy of current parameters to compute magnitude of updates,
            # and copy them to CPU so large models won't go OOM on the GPU.
            self.param_updates = {name: param.detach().cpu().clone()
                                  for name, param in state.model.named_parameters()}

        elif event == Events.BATCH_END:
            # Log parameter values to tensorboard
            if self.tensorboard.should_log_this_batch():
                self.tensorboard.log_parameter_and_gradient_statistics(state.model, state.batch_grad_norm)
                self.tensorboard.log_learning_rates(state.model, state.optimizer)

                self.tensorboard.add_train_scalar("loss/loss_train", state.train_metrics["loss"])
                self.tensorboard.log_metrics({"epoch_metrics/" + k: v for k, v in state.train_metrics.items()})


            if self.log_batch_size_period:
                cur_batch = sum([training_util.get_batch_size(batch) for batch in state.batch_group])
                self.cumulative_batch_size += cur_batch
                if (state.batches_this_epoch - 1) % self.log_batch_size_period == 0:
                    average = self.cumulative_batch_size / state.batches_this_epoch
                    logger.info(f"current batch size: {cur_batch} mean batch size: {average}")
                    self.tensorboard.add_train_scalar("current_batch_size", cur_batch)
                    self.tensorboard.add_train_scalar("mean_batch_size", average)

            if self.tensorboard.should_log_histograms_this_batch():
                for name, param in state.model.named_parameters():
                    self.param_updates[name].sub_(param.detach().cpu())
                    update_norm = torch.norm(self.param_updates[name].view(-1, ))
                    param_norm = torch.norm(param.view(-1, )).cpu()
                    self.tensorboard.add_train_scalar("gradient_update/" + name,
                                                      update_norm / (param_norm + 1e-7))
                self.param_updates.clear()
                self.tensorboard.log_histograms(state.model, self.histogram_parameters)

        elif event == Events.EPOCH_END:
            self.tensorboard.log_metrics(state.train_metrics,
                                         val_metrics=state.val_metrics,
                                         log_to_console=True,
                                         epoch=self.epoch)
            self.epoch += 1

    @classmethod
    def from_params(cls, serialization_dir: str, params: Params) -> 'LogTensorboard':
        log_batch_size_period = params.pop_int("log_batch_size_period", None)
        tensorboard = TensorboardWriter.from_params(params=params,
                                                    serialization_dir=serialization_dir,
                                                    get_batch_num_total=lambda: None)
        return LogTensorboard(tensorboard, log_batch_size_period)


@Callback.register("learning_rate_scheduler")
class LrsCallback(Callback['trainer2.Trainer']):
    def __init__(self, learning_rate_scheduler: LearningRateScheduler) -> None:
        self.learning_rate_scheduler = learning_rate_scheduler

    def __call__(self, event: str, state: 'trainer2.Trainer') -> None:
        # Don't do anything if there's no lr_scheduler
        if self.learning_rate_scheduler is None:
            return

        if event == Events.AFTER_BACKWARD:
            self.learning_rate_scheduler.step_batch(state.batch_num_total)
        elif event == Events.EPOCH_END:
            self.learning_rate_scheduler.step(state.latest_val_metric, state.epoch_number)

    def get_training_state(self) -> dict:
        return {"learning_rate_scheduler": self.learning_rate_scheduler.state_dict()}

    def restore_training_state(self, training_state: dict) -> None:
        state_dict = training_state.pop("learning_rate_scheduler", None)

        if state_dict:
            self.learning_rate_scheduler.load_state_dict(state_dict)



    @classmethod
    def from_params(cls, params: Params, optimizer: Optimizer) -> 'LrsCallback':
        learning_rate_scheduler = LearningRateScheduler.from_params(params=params.pop("learning_rate_scheduler"),
                                                                    optimizer=optimizer)
        return LrsCallback(learning_rate_scheduler)


@Callback.register("momentum_scheduler")
class MomentumSchedulerCallback(Callback['trainer2.Trainer']):
    def __init__(self, momentum_scheduler: MomentumScheduler) -> None:
        self.momentum_scheduler = momentum_scheduler

    def __call__(self, event: str, state: 'trainer2.Trainer') -> None:
        # Don't do anything if there's no momentum_scheduler
        if self.momentum_scheduler is None:
            return

        if event == Events.AFTER_BACKWARD:
            self.momentum_scheduler.step_batch(state.batch_num_total)
        elif event == Events.EPOCH_END:
            self.momentum_scheduler.step(state.latest_val_metric, state.epoch_number)

    def get_training_state(self) -> dict:
        return {"momentum_scheduler": self.momentum_scheduler.state_dict()}

    def restore_training_state(self, training_state: dict) -> None:
        state_dict = training_state.pop("momentum_scheduler", None)

        if state_dict:
            self.momentum_scheduler.load_state_dict(state_dict)

    @classmethod
    def from_params(cls, params: Params, optimizer: Optimizer) -> 'LrsCallback':
        learning_rate_scheduler = LearningRateScheduler.from_params(params=params.pop("learning_rate_scheduler"),
                                                                    optimizer=optimizer)
        return LrsCallback(learning_rate_scheduler)


_DEFAULT_STATE_DICT_ATTRS = ['optimizer']

_DEFAULT_OTHER_ATTRS = ['batch_num_total']


@Callback.register("checkpoint")
class CheckpointCallback(Callback['trainer2.Trainer']):
    def __init__(self,
                 checkpointer: Checkpointer,
                 state_dict_attrs: List[str] = None,
                 other_attrs: List[str] = None) -> None:
        self.checkpointer = checkpointer
        self.state_dict_attrs = state_dict_attrs or _DEFAULT_STATE_DICT_ATTRS
        self.other_attrs = other_attrs or _DEFAULT_OTHER_ATTRS

    def __call__(self, event: str, state: 'trainer2.Trainer') -> None:
        if event == Events.SAVE_CHECKPOINT:
            training_states = {}

            # Add state_dict attributes
            for attr in self.state_dict_attrs:
                state_attr = getattr(state, attr)
                if state_attr is not None:
                    training_states[attr] = state_attr.state_dict()

            # Add other attributes
            for attr in self.other_attrs:
                training_states[attr] = getattr(state, attr)

            # Get attributes from callbacks
            for callback in state.handler.callbacks:
                training_states.update(callback.get_training_state())

            is_best_so_far = training_states.pop("is_best_so_far", True)
            self.checkpointer.save_checkpoint(
                    model_state=state.model.state_dict(),
                    epoch=state.checkpoint_epoch,
                    training_states=training_states,
                    is_best_so_far=is_best_so_far)


        elif event == Events.RESTORE_CHECKPOINT:
            # Restores the model and training state from the last saved checkpoint.
            # This includes an epoch count and optimizer state, which is serialized separately
            # from model parameters. This function should only be used to continue training -
            # if you wish to load a model for inference/load parts of a model into a new
            # computation graph, you should use the native Pytorch functions:
            # `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

            # If ``self._serialization_dir`` does not exist or does not contain any checkpointed weights,
            # this will do nothing.
            try:
                model_state, training_state = self.checkpointer.restore_checkpoint()
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

            # Restore state_dict attrs
            for attr in self.state_dict_attrs:
                state_attr = getattr(state, attr)
                if state_attr is not None:
                    state_attr.load_state_dict(training_state[attr])

            # Restore other attrs
            for attr in self.other_attrs:
                setattr(state, attr, training_state[attr])

            # Restore callback attrs
            for callback in state.handler.callbacks:
                callback.restore_training_state(training_state)

            if isinstance(training_state["epoch"], int):
                state.epoch_number = training_state["epoch"] + 1
            else:
                state.epoch_number = int(training_state["epoch"].split('.')[0]) + 1

        elif event == Events.TRAINING_END:
            # Load the best model state before returning
            best_model_state = self.checkpointer.best_model_state()
            if best_model_state:
                state.model.load_state_dict(best_model_state)

    @classmethod
    def from_params(cls, params: Params, serialization_dir: str) -> 'CheckpointCallback':
        checkpointer_params = params.pop("checkpointer", None)
        if checkpointer_params:
            checkpointer = Checkpointer.from_params(checkpointer_params, serialization_dir=serialization_dir)
        else:
            checkpointer = Checkpointer(serialization_dir=serialization_dir)

        state_dict_attrs = params.pop("state_dict_attrs", None)
        other_attrs = params.pop("other_attrs", None)

        return CheckpointCallback(checkpointer, state_dict_attrs, other_attrs)


@Callback.register("moving_average")
class MovingAverageCallback(Callback['trainer2.Trainer']):
    def __init__(self, moving_average: MovingAverage) -> None:
        self.moving_average = moving_average

    def __call__(self, event: str, state: 'trainer2.Trainer') -> None:
        if self.moving_average is None:
            return

        if event == Events.BATCH_END:
            self.moving_average.apply(state.batch_num_total)

        elif event in [Events.BEFORE_SAVE_CHECKPOINT, Events.BEFORE_VALIDATE]:
            # If moving averages are used for parameters, we save
            # the moving average values into checkpoint, instead of the current values.
            self.moving_average.assign_average_value()

        elif event in [Events.AFTER_SAVE_CHECKPOINT, Events.AFTER_VALIDATE]:
            # Restore the original values for parameters so that training will not be affected.
            self.moving_average.restore()

    @classmethod
    def from_params(cls, params: Params, model: Model) -> 'MovingAverageCallback':
        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        moving_average = MovingAverage.from_params(params.pop("moving_average"), parameters=parameters)
        return MovingAverageCallback(moving_average)



@Callback.register("validate")
class Validate(Callback['trainer2.Trainer']):
    def __call__(self, event: str, state: 'trainer2.Trainer') -> None:
        if event == Events.VALIDATE and state.validation_data is not None:

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

@Callback.register("track_metrics")
class TrackMetrics(Callback['trainer2.Trainer']):
    # We want this to happen last, generally.

    priority = 100
    def __init__(self,
                 patience: int = None,
                 validation_metric: str = "-loss") -> None:
        if patience is not None and (not isinstance(patience, int) or patience <= 0):
            raise ConfigurationError(f"patience must be a positive number, but got {patience}."
                                     f"To disable early stopping, don't specify it.")

        self.patience = patience
        self.validation_metric = validation_metric[1:]
        self.metric_tracker = MetricTracker(patience, validation_metric)
        self.starting_epoch = 0

        self.peak_cpu_usage = 0
        # Track pairs (gpu_id, memory usage)
        self.gpu_usage: List[Tuple[int, int]] = []


    def get_training_state(self) -> dict:
        return {
                "metric_tracker": self.metric_tracker.state_dict(),
                # This is already in the metric_tracker state dict, but it makes our lives easier.
                "is_best_so_far": self.metric_tracker.is_best_so_far()
        }

    def restore_training_state(self, training_state: dict) -> None:
        state_dict = training_state.pop("metric_tracker", None)

        if state_dict:
            self.metric_tracker.load_state_dict(state_dict)

    def __call__(self, event: str, state: 'trainer2.Trainer') -> None:
        if event == Events.TRAINING_START:
            # Keep track of starting epoch
            self.starting_epoch = state.epoch_number

            if self.patience is None and state.validation_data is not None:
                logger.warning('You provided a validation dataset but patience was set to None, '
                               'meaning that early stopping is disabled')

            state.metrics['best_epoch'] = self.metric_tracker.best_epoch or 0
            for key, value in self.metric_tracker.best_epoch_metrics.items():
                state.metrics["best_validation_" + key] = value

        elif event == Events.EPOCH_START:
            # This used to be in train_epoch()
            logger.info("Epoch %d/%d", state.epoch_number, state.num_epochs - 1)
            self.peak_cpu_usage = peak_memory_mb()
            logger.info(f"Peak CPU memory usage MB: {self.peak_cpu_usage}")
            self.gpu_usage.clear()
            for gpu, memory in gpu_memory_mb().items():
                self.gpu_usage.append((gpu, memory))
                logger.info(f"GPU {gpu} memory usage MB: {memory}")


        elif event == Events.VALIDATE:
            state.train_metrics = training_util.get_metrics(state.model,
                                                            state.train_loss,
                                                            state.batches_this_epoch,
                                                            reset=True)
            state.train_metrics['cpu_memory_MB'] = self.peak_cpu_usage
            for (gpu_num, memory) in self.gpu_usage:
                state.train_metrics['gpu_'+str(gpu_num)+'_memory_MB'] = memory

            # get peak of memory usage
            if 'cpu_memory_MB' in state.train_metrics:
                state.metrics['peak_cpu_memory_MB'] = max(state.metrics.get('peak_cpu_memory_MB', 0),
                                                          state.train_metrics['cpu_memory_MB'])
            for key, value in state.train_metrics.items():
                if key.startswith('gpu_'):
                    state.metrics["peak_"+key] = max(state.metrics.get("peak_"+key, 0), value)

            if state.validation_data is not None:
                # Check validation metric for early stopping
                state.latest_val_metric = state.val_metrics[self.validation_metric]
                self.metric_tracker.add_metric(state.latest_val_metric)

                if self.metric_tracker.should_stop_early():
                    state.should_stop_early = True

        elif event == Events.EPOCH_END:
            # Create overall metrics dict
            training_elapsed_time = time.time() - state.training_start_time
            state.metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
            state.metrics["training_start_epoch"] = self.starting_epoch
            state.metrics["training_epochs"] = state.epoch_number - self.starting_epoch + 1
            state.metrics["epoch"] = state.epoch_number

            for key, value in state.train_metrics.items():
                state.metrics["training_" + key] = value
            for key, value in state.val_metrics.items():
                state.metrics["validation_" + key] = value

            if self.metric_tracker.is_best_so_far():
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                state.metrics['best_epoch'] = state.epoch_number
                for key, value in state.val_metrics.items():
                    state.metrics["best_validation_" + key] = value

                self.metric_tracker.best_epoch_metrics = copy.deepcopy(state.val_metrics)

            # pylint: disable=protected-access
            if state._serialization_dir:
                dump_metrics(os.path.join(state._serialization_dir, f'metrics_epoch_{state.epoch_number}.json'),
                             state.metrics)
