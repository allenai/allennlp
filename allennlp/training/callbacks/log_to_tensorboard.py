# These should probably all live in separate files
from typing import Set, Dict, TYPE_CHECKING
import logging

import torch

from allennlp.common.params import Params
from allennlp.training import util as training_util
from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.training.callbacks.events import Events
from allennlp.training.tensorboard_writer import TensorboardWriter

if TYPE_CHECKING:
    from allennlp.training.callback_trainer import CallbackTrainer

logger = logging.getLogger(__name__)


@Callback.register("log_to_tensorboard")
class LogToTensorboard(Callback):
    """
    Callback that handles all Tensorboard logging.

    # Parameters

    tensorboard : `TensorboardWriter`
        The TensorboardWriter instance to write to.
    log_batch_size_period : int, optional (default: None)
        If provided, we'll log the average batch sizes to Tensorboard
        every this-many batches.
    """

    def __init__(self, tensorboard: TensorboardWriter, log_batch_size_period: int = None) -> None:
        self.log_batch_size_period = log_batch_size_period
        self.tensorboard = tensorboard

        self.cumulative_batch_size = 0

        # For logging histograms
        self.histogram_parameters: Set[str] = set()
        self.param_updates: Dict[str, torch.Tensor] = {}

    @handle_event(Events.TRAINING_START)
    def training_start(self, trainer: "CallbackTrainer"):
        # This is an ugly hack to get the tensorboard instance to know about the trainer, because
        # the callbacks are defined before the trainer.
        # TODO: figure out a better way to handle this.
        self.tensorboard._get_batch_num_total = lambda: trainer.batch_num_total

        # Get histogram parameters
        self.histogram_parameters = set(
            trainer.model.get_parameters_for_histogram_tensorboard_logging()
        )

        # Enable activation logging.
        if self.tensorboard._histogram_interval is not None:
            self.tensorboard.enable_activation_logging(trainer.model)

    @handle_event(Events.BATCH_START)
    def copy_current_parameters(self, trainer: "CallbackTrainer"):
        if self.tensorboard.should_log_histograms_this_batch():
            # Get the magnitude of parameter updates for logging
            # We need a copy of current parameters to compute magnitude of updates,
            # and copy them to CPU so large models won't go OOM on the GPU.
            self.param_updates = {
                name: param.detach().cpu().clone()
                for name, param in trainer.model.named_parameters()
            }

    @handle_event(Events.BATCH_END)
    def batch_end_logging(self, trainer: "CallbackTrainer"):
        # Log parameter values to tensorboard
        if self.tensorboard.should_log_this_batch():
            self.tensorboard.log_parameter_and_gradient_statistics(
                trainer.model, trainer.batch_grad_norm
            )
            self.tensorboard.log_learning_rates(trainer.model, trainer.optimizer)

            self.tensorboard.add_train_scalar("loss/loss_train", trainer.train_metrics["loss"])
            self.tensorboard.log_metrics(
                {"epoch_metrics/" + k: v for k, v in trainer.train_metrics.items()}
            )

        if self.log_batch_size_period:
            cur_batch = training_util.get_batch_size(trainer.batch)
            self.cumulative_batch_size += cur_batch
            if (trainer.batches_this_epoch - 1) % self.log_batch_size_period == 0:
                average = self.cumulative_batch_size / trainer.batches_this_epoch
                logger.debug(f"current batch size: {cur_batch} mean batch size: {average}")
                self.tensorboard.add_train_scalar("current_batch_size", cur_batch)
                self.tensorboard.add_train_scalar("mean_batch_size", average)

        if self.tensorboard.should_log_histograms_this_batch():
            for name, param in trainer.model.named_parameters():
                self.param_updates[name].sub_(param.detach().cpu())
                update_norm = torch.norm(self.param_updates[name].view(-1))
                param_norm = torch.norm(param.view(-1)).cpu()
                self.tensorboard.add_train_scalar(
                    "gradient_update/" + name, update_norm / (param_norm + 1e-7)
                )
            self.param_updates.clear()
            self.tensorboard.log_histograms(trainer.model, self.histogram_parameters)

    @handle_event(Events.EPOCH_END)
    def epoch_end_logging(self, trainer: "CallbackTrainer"):
        self.tensorboard.log_metrics(
            trainer.train_metrics,
            val_metrics=trainer.val_metrics,
            log_to_console=True,
            epoch=trainer.epoch_number + 1,
        )

    @handle_event(Events.TRAINING_END)
    def training_end(self, trainer: "CallbackTrainer"):

        self.tensorboard.close()

    @classmethod
    def from_params(  # type: ignore
        cls, serialization_dir: str, params: Params, **extras
    ) -> "LogToTensorboard":
        log_batch_size_period = params.pop_int("log_batch_size_period", None)
        tensorboard = TensorboardWriter.from_params(
            params=params, serialization_dir=serialization_dir, get_batch_num_total=lambda: None
        )
        # TODO(mattg): remove get_batch_num_total from TensorboardWriter, and instead just add a
        # method / arguments to tell the writer what batch num we're at.
        return LogToTensorboard(tensorboard, log_batch_size_period)
