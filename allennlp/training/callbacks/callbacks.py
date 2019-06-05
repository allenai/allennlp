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

    def __call__(self, event: str, trainer: 'trainer2.Trainer') -> None:
        if event == Events.TRAINING_START:
            # Bad hack to get the tensorboard instance to know about the trainer
            # pylint: disable=protected-access
            self.tensorboard._get_batch_num_total = lambda: trainer.batch_num_total

            # Get histogram parameters
            self.histogram_parameters = set(
                    trainer.model.get_parameters_for_histogram_tensorboard_logging()
            )

            # Enable activation logging.
            if self.tensorboard._histogram_interval is not None:
                self.tensorboard.enable_activation_logging(trainer.model)

        elif event == Events.BATCH_START and self.tensorboard.should_log_histograms_this_batch():
            # get the magnitude of parameter updates for logging
            # We need a copy of current parameters to compute magnitude of updates,
            # and copy them to CPU so large models won't go OOM on the GPU.
            self.param_updates = {name: param.detach().cpu().clone()
                                  for name, param in trainer.model.named_parameters()}

        elif event == Events.BATCH_END:
            # Log parameter values to tensorboard
            if self.tensorboard.should_log_this_batch():
                self.tensorboard.log_parameter_and_gradient_statistics(trainer.model, trainer.batch_grad_norm)
                self.tensorboard.log_learning_rates(trainer.model, trainer.optimizer)

                self.tensorboard.add_train_scalar("loss/loss_train", trainer.train_metrics["loss"])
                self.tensorboard.log_metrics({"epoch_metrics/" + k: v for k, v in trainer.train_metrics.items()})


            if self.log_batch_size_period:
                cur_batch = sum([training_util.get_batch_size(batch) for batch in trainer.batch_group])
                self.cumulative_batch_size += cur_batch
                if (trainer.batches_this_epoch - 1) % self.log_batch_size_period == 0:
                    average = self.cumulative_batch_size / trainer.batches_this_epoch
                    logger.info(f"current batch size: {cur_batch} mean batch size: {average}")
                    self.tensorboard.add_train_scalar("current_batch_size", cur_batch)
                    self.tensorboard.add_train_scalar("mean_batch_size", average)

            if self.tensorboard.should_log_histograms_this_batch():
                for name, param in trainer.model.named_parameters():
                    self.param_updates[name].sub_(param.detach().cpu())
                    update_norm = torch.norm(self.param_updates[name].view(-1, ))
                    param_norm = torch.norm(param.view(-1, )).cpu()
                    self.tensorboard.add_train_scalar("gradient_update/" + name,
                                                      update_norm / (param_norm + 1e-7))
                self.param_updates.clear()
                self.tensorboard.log_histograms(trainer.model, self.histogram_parameters)

        elif event == Events.EPOCH_END:
            self.tensorboard.log_metrics(trainer.train_metrics,
                                         val_metrics=trainer.val_metrics,
                                         log_to_console=True,
                                         epoch=self.epoch)
            self.epoch += 1

    @classmethod
    def from_params(cls, serialization_dir: str, params: Params) -> 'LogTensorboard':  # type: ignore
        log_batch_size_period = params.pop_int("log_batch_size_period", None)
        tensorboard = TensorboardWriter.from_params(params=params,
                                                    serialization_dir=serialization_dir,
                                                    get_batch_num_total=lambda: None)
        return LogTensorboard(tensorboard, log_batch_size_period)


@Callback.register("learning_rate_scheduler")
class LrsCallback(Callback['trainer2.Trainer']):
    def __init__(self, learning_rate_scheduler: LearningRateScheduler) -> None:
        self.learning_rate_scheduler = learning_rate_scheduler

    def __call__(self, event: str, trainer: 'trainer2.Trainer') -> None:
        # Don't do anything if there's no lr_scheduler
        if self.learning_rate_scheduler is None:
            return

        if event == Events.AFTER_BACKWARD:
            self.learning_rate_scheduler.step_batch(trainer.batch_num_total)
        elif event == Events.EPOCH_END:
            self.learning_rate_scheduler.step(trainer.latest_val_metric, trainer.epoch_number)

    def get_training_state(self) -> dict:
        return {"learning_rate_scheduler": self.learning_rate_scheduler.state_dict()}

    def restore_training_state(self, training_trainer: dict) -> None:
        state_dict = training_trainer.pop("learning_rate_scheduler", None)

        if state_dict:
            self.learning_rate_scheduler.load_state_dict(state_dict)



    @classmethod
    def from_params(cls, params: Params, optimizer: Optimizer) -> 'LrsCallback':  # type: ignore
        learning_rate_scheduler = LearningRateScheduler.from_params(params=params.pop("learning_rate_scheduler"),
                                                                    optimizer=optimizer)
        return LrsCallback(learning_rate_scheduler)


@Callback.register("momentum_scheduler")
class MomentumSchedulerCallback(Callback['trainer2.Trainer']):
    def __init__(self, momentum_scheduler: MomentumScheduler) -> None:
        self.momentum_scheduler = momentum_scheduler

    def __call__(self, event: str, trainer: 'trainer2.Trainer') -> None:
        # Don't do anything if there's no momentum_scheduler
        if self.momentum_scheduler is None:
            return

        if event == Events.AFTER_BACKWARD:
            self.momentum_scheduler.step_batch(trainer.batch_num_total)
        elif event == Events.EPOCH_END:
            self.momentum_scheduler.step(trainer.latest_val_metric, trainer.epoch_number)

    def get_training_state(self) -> dict:
        return {"momentum_scheduler": self.momentum_scheduler.state_dict()}

    def restore_training_state(self, training_trainer: dict) -> None:
        state_dict = training_trainer.pop("momentum_scheduler", None)

        if state_dict:
            self.momentum_scheduler.load_state_dict(state_dict)

    @classmethod
    def from_params(cls, params: Params, optimizer: Optimizer) -> 'LrsCallback':  # type: ignore
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

    def __call__(self, event: str, trainer: 'trainer2.Trainer') -> None:
        if event == Events.SAVE_CHECKPOINT:
            training_states = {}

            # Add state_dict attributes
            for attr in self.state_dict_attrs:
                state_attr = getattr(trainer, attr)
                if state_attr is not None:
                    training_states[attr] = state_attr.state_dict()

            # Add other attributes
            for attr in self.other_attrs:
                training_states[attr] = getattr(trainer, attr)

            # Get attributes from callbacks
            for callback in trainer.handler.callbacks:
                training_states.update(callback.get_training_state())

            is_best_so_far = training_states.pop("is_best_so_far", True)
            self.checkpointer.save_checkpoint(
                    model_state=trainer.model.state_dict(),
                    epoch=trainer.checkpoint_epoch,
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
                trainer.epoch_number = 0
                return

            trainer.model.load_state_dict(model_state)

            # Restore state_dict attrs
            for attr in self.state_dict_attrs:
                state_attr = getattr(trainer, attr)
                if state_attr is not None:
                    state_attr.load_state_dict(training_state[attr])

            # Restore other attrs
            for attr in self.other_attrs:
                setattr(trainer, attr, training_state[attr])

            # Restore callback attrs
            for callback in trainer.handler.callbacks:
                callback.restore_training_state(training_state)

            if isinstance(training_state["epoch"], int):
                trainer.epoch_number = training_state["epoch"] + 1
            else:
                trainer.epoch_number = int(training_state["epoch"].split('.')[0]) + 1

        elif event == Events.TRAINING_END:
            # Load the best model state before returning
            best_model_state = self.checkpointer.best_model_state()
            if best_model_state:
                trainer.model.load_state_dict(best_model_state)

    @classmethod
    def from_params(cls, params: Params, serialization_dir: str) -> 'CheckpointCallback':  # type: ignore
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

    def __call__(self, event: str, trainer: 'trainer2.Trainer') -> None:
        if self.moving_average is None:
            return

        if event == Events.BATCH_END:
            self.moving_average.apply(trainer.batch_num_total)

        elif event in [Events.BEFORE_SAVE_CHECKPOINT, Events.BEFORE_VALIDATE]:
            # If moving averages are used for parameters, we save
            # the moving average values into checkpoint, instead of the current values.
            self.moving_average.assign_average_value()

        elif event in [Events.AFTER_SAVE_CHECKPOINT, Events.AFTER_VALIDATE]:
            # Restore the original values for parameters so that training will not be affected.
            self.moving_average.restore()

    @classmethod
    def from_params(cls, params: Params, model: Model) -> 'MovingAverageCallback':  # type: ignore
        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        moving_average = MovingAverage.from_params(params.pop("moving_average"), parameters=parameters)
        return MovingAverageCallback(moving_average)



@Callback.register("validate")
class Validate(Callback['trainer2.Trainer']):
    def __call__(self, event: str, trainer: 'trainer2.Trainer') -> None:
        if event == Events.VALIDATE and trainer.validation_data is not None:

            with torch.no_grad():
                # We have a validation set, so compute all the metrics on it.
                logger.info("Validating")

                trainer.model.eval()

                num_gpus = len(trainer._cuda_devices)  # pylint: disable=protected-access

                raw_val_generator = trainer.validation_iterator(
                        trainer.validation_data,
                        num_epochs=1,
                        shuffle=False)
                val_generator = lazy_groups_of(raw_val_generator, num_gpus)
                num_validation_batches = math.ceil(
                        trainer.validation_iterator.get_num_batches(trainer.validation_data) / num_gpus)
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

        self.peak_cpu_usage = 0.0
        # Track pairs (gpu_id, memory usage)
        self.gpu_usage: List[Tuple[int, int]] = []


    def get_training_state(self) -> dict:
        return {
                "metric_tracker": self.metric_tracker.state_dict(),
                # This is already in the metric_tracker state dict, but it makes our lives easier.
                "is_best_so_far": self.metric_tracker.is_best_so_far()
        }

    def restore_training_state(self, training_trainer: dict) -> None:
        state_dict = training_trainer.pop("metric_tracker", None)

        if state_dict:
            self.metric_tracker.load_state_dict(state_dict)

    def __call__(self, event: str, trainer: 'trainer2.Trainer') -> None:
        if event == Events.TRAINING_START:
            # Keep track of starting epoch
            self.starting_epoch = trainer.epoch_number

            if self.patience is None and trainer.validation_data is not None:
                logger.warning('You provided a validation dataset but patience was set to None, '
                               'meaning that early stopping is disabled')

            trainer.metrics['best_epoch'] = self.metric_tracker.best_epoch or 0
            for key, value in self.metric_tracker.best_epoch_metrics.items():
                trainer.metrics["best_validation_" + key] = value

        elif event == Events.EPOCH_START:
            # This used to be in train_epoch()
            logger.info("Epoch %d/%d", trainer.epoch_number, trainer.num_epochs - 1)
            self.peak_cpu_usage = peak_memory_mb()
            logger.info(f"Peak CPU memory usage MB: {self.peak_cpu_usage}")
            self.gpu_usage.clear()
            for gpu, memory in gpu_memory_mb().items():
                self.gpu_usage.append((gpu, memory))
                logger.info(f"GPU {gpu} memory usage MB: {memory}")


        elif event == Events.VALIDATE:
            trainer.train_metrics = training_util.get_metrics(trainer.model,
                                                              trainer.train_loss,
                                                              trainer.batches_this_epoch,
                                                              reset=True)
            trainer.train_metrics['cpu_memory_MB'] = self.peak_cpu_usage
            for (gpu_num, memory) in self.gpu_usage:
                trainer.train_metrics['gpu_'+str(gpu_num)+'_memory_MB'] = memory

            # get peak of memory usage
            if 'cpu_memory_MB' in trainer.train_metrics:
                trainer.metrics['peak_cpu_memory_MB'] = max(trainer.metrics.get('peak_cpu_memory_MB', 0),
                                                            trainer.train_metrics['cpu_memory_MB'])
            for key, value in trainer.train_metrics.items():
                if key.startswith('gpu_'):
                    trainer.metrics["peak_"+key] = max(trainer.metrics.get("peak_"+key, 0), value)

            if trainer.validation_data is not None:
                # Check validation metric for early stopping
                trainer.latest_val_metric = trainer.val_metrics[self.validation_metric]
                self.metric_tracker.add_metric(trainer.latest_val_metric)

                if self.metric_tracker.should_stop_early():
                    trainer.should_stop_early = True

        elif event == Events.EPOCH_END:
            # Create overall metrics dict
            training_elapsed_time = time.time() - trainer.training_start_time
            trainer.metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
            trainer.metrics["training_start_epoch"] = self.starting_epoch
            trainer.metrics["training_epochs"] = trainer.epoch_number - self.starting_epoch + 1
            trainer.metrics["epoch"] = trainer.epoch_number

            for key, value in trainer.train_metrics.items():
                trainer.metrics["training_" + key] = value
            for key, value in trainer.val_metrics.items():
                trainer.metrics["validation_" + key] = value

            if self.metric_tracker.is_best_so_far():
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                trainer.metrics['best_epoch'] = trainer.epoch_number
                for key, value in trainer.val_metrics.items():
                    trainer.metrics["best_validation_" + key] = value

                self.metric_tracker.best_epoch_metrics = copy.deepcopy(trainer.val_metrics)

            # pylint: disable=protected-access
            if trainer._serialization_dir:
                dump_metrics(os.path.join(trainer._serialization_dir,
                                          f'metrics_epoch_{trainer.epoch_number}.json'),
                             trainer.metrics)
