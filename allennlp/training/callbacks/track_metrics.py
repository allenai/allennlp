from typing import List, Tuple, TYPE_CHECKING
import copy
import datetime
import logging
import os
import time

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import dump_metrics, gpu_memory_mb, peak_memory_mb
from allennlp.training import util as training_util
from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.training.callbacks.events import Events
from allennlp.training.metric_tracker import MetricTracker

if TYPE_CHECKING:
    from allennlp.training.callback_trainer import CallbackTrainer

logger = logging.getLogger(__name__)


@Callback.register("track_metrics")
class TrackMetrics(Callback):
    """
    Callback that handles tracking of metrics and (potentially) early stopping.

    # Parameters

    patience : int, optional (default = None)
        If a positive number is provided, training will stop when the supplied
        validation_metric has not improved in this many epochs.
    validation_metric : str, optional (default = "-loss")
        The metric to use for early stopping. The initial +/- indicates whether
        we expect the metric to increase or decrease during training.
    """

    def __init__(self, patience: int = None, validation_metric: str = "-loss") -> None:
        if patience is not None and (not isinstance(patience, int) or patience <= 0):
            raise ConfigurationError(
                f"patience must be a positive number, but got {patience}."
                f"To disable early stopping, don't specify it."
            )

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
            "is_best_so_far": self.metric_tracker.is_best_so_far(),
        }

    def restore_training_state(self, training_state: dict) -> None:
        state_dict = training_state.pop("metric_tracker", None)

        if state_dict:
            self.metric_tracker.load_state_dict(state_dict)

    @handle_event(Events.TRAINING_START, priority=100)
    def set_up_metrics(self, trainer: "CallbackTrainer"):
        # Keep track of starting epoch
        self.starting_epoch = trainer.epoch_number

        if self.patience is None and trainer.validate:
            logger.warning(
                "You provided a validation dataset but patience was set to None, "
                "meaning that early stopping is disabled"
            )

        trainer.metrics["best_epoch"] = self.metric_tracker.best_epoch or 0
        for key, value in self.metric_tracker.best_epoch_metrics.items():
            trainer.metrics["best_validation_" + key] = value

    @handle_event(Events.EPOCH_START, priority=100)
    def measure_cpu_gpu(self, trainer: "CallbackTrainer"):
        # This used to be in train_epoch()
        logger.info("Epoch %d/%d", trainer.epoch_number, trainer.num_epochs - 1)
        self.peak_cpu_usage = peak_memory_mb()
        logger.info(f"Peak CPU memory usage MB: {self.peak_cpu_usage}")
        self.gpu_usage.clear()
        for gpu, memory in gpu_memory_mb().items():
            self.gpu_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

    # We want to collect training metrics before the actual validation happens
    @handle_event(Events.VALIDATE, priority=-100)
    def collect_train_metrics(self, trainer: "CallbackTrainer"):
        trainer.train_metrics = training_util.get_metrics(
            trainer.model, trainer.train_loss, trainer.batches_this_epoch, reset=True
        )
        trainer.train_metrics["cpu_memory_MB"] = self.peak_cpu_usage
        for (gpu_num, memory) in self.gpu_usage:
            trainer.train_metrics["gpu_" + str(gpu_num) + "_memory_MB"] = memory

        # get peak of memory usage
        if "cpu_memory_MB" in trainer.train_metrics:
            trainer.metrics["peak_cpu_memory_MB"] = max(
                trainer.metrics.get("peak_cpu_memory_MB", 0), trainer.train_metrics["cpu_memory_MB"]
            )
        for key, value in trainer.train_metrics.items():
            if key.startswith("gpu_"):
                trainer.metrics["peak_" + key] = max(trainer.metrics.get("peak_" + key, 0), value)

    # We want to collect validation metrics after the validation happens
    @handle_event(Events.VALIDATE, priority=100)
    def collect_val_metrics(self, trainer: "CallbackTrainer"):
        if trainer.validate:
            # Check validation metric for early stopping
            trainer.latest_val_metric = trainer.val_metrics[self.validation_metric]
            self.metric_tracker.add_metric(trainer.latest_val_metric)

            if self.metric_tracker.should_stop_early():
                trainer.should_stop_early = True

    @handle_event(Events.EPOCH_END, priority=100)
    def end_of_epoch(self, trainer: "CallbackTrainer"):
        # Create overall metrics dict
        training_elapsed_time = time.time() - trainer.training_start_time
        trainer.metrics["training_duration"] = str(
            datetime.timedelta(seconds=training_elapsed_time)
        )
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
            trainer.metrics["best_epoch"] = trainer.epoch_number
            for key, value in trainer.val_metrics.items():
                trainer.metrics["best_validation_" + key] = value

            self.metric_tracker.best_epoch_metrics = copy.deepcopy(trainer.val_metrics)

        if trainer._serialization_dir:
            dump_metrics(
                os.path.join(
                    trainer._serialization_dir, f"metrics_epoch_{trainer.epoch_number}.json"
                ),
                trainer.metrics,
            )
