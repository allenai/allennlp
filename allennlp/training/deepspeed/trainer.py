import datetime
import logging
import os
import re
import time
import traceback
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from overrides import overrides

import torch
import torch.distributed as dist

from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.utils import logger as ds_logger

from allennlp.common import Lazy, Tqdm
from allennlp.common import util as common_util
from allennlp.common.checks import ConfigurationError
from allennlp.data import DataLoader
from allennlp.data.dataloader import TensorDict
from allennlp.models.model import Model
from allennlp.nn import util as nn_util
from allennlp.training import util as training_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer
from allennlp.training.tensorboard_writer import TensorboardWriter

from allennlp.training.trainer import (
    Trainer,
    GradientDescentTrainer,
    BatchCallback,
    EpochCallback,
    TrainerCallback,
)

from allennlp.training.deepspeed.config import DeepspeedConfig, DeepspeedArgs
from allennlp.training.deepspeed.checkpointer import DeepspeedCheckpointer

logger = logging.getLogger(__name__)
ds_logger.setLevel(logging.WARNING)
ds_logger.propagate = False


@Trainer.register("deepspeed", constructor="from_partial_objects")
class DeepspeedTrainer(GradientDescentTrainer):
    def __init__(
        self,
        model: Model,
        data_loader: DataLoader,
        deepspeed_engine: DeepSpeedEngine,
        patience: Optional[int] = None,
        validation_metric: str = "-loss",
        validation_data_loader: DataLoader = None,
        num_epochs: int = 20,
        serialization_dir: Optional[str] = None,
        checkpointer: Checkpointer = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        tensorboard_writer: TensorboardWriter = None,
        moving_average: Optional[MovingAverage] = None,
        batch_callbacks: List[BatchCallback] = None,
        epoch_callbacks: List[EpochCallback] = None,
        end_callbacks: List[EpochCallback] = None,
        trainer_callbacks: List[TrainerCallback] = None,
        distributed: bool = False,
        local_rank: int = 0,
        world_size: int = 1,
        num_gradient_accumulation_steps: int = 1,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=deepspeed_engine.optimizer,
            data_loader=data_loader,
            patience=patience,
            validation_metric=validation_metric,
            validation_data_loader=validation_data_loader,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            tensorboard_writer=tensorboard_writer,
            checkpointer=checkpointer,
            moving_average=moving_average,
            batch_callbacks=batch_callbacks,
            epoch_callbacks=epoch_callbacks,
            end_callbacks=end_callbacks,
            trainer_callbacks=trainer_callbacks,
            distributed=False,
            local_rank=local_rank,
            world_size=world_size,
            num_gradient_accumulation_steps=num_gradient_accumulation_steps,
            use_amp=False,
        )

        self.model_engine = deepspeed_engine
        self._distributed = True

        if checkpointer is None and serialization_dir is not None:
            self._checkpointer = DeepspeedCheckpointer(serialization_dir)

    def batch_outputs(self, batch: TensorDict, for_training: bool) -> Dict[str, torch.Tensor]:
        """
        Does a forward pass on the given batch and returns the output dictionary that the model
        returns, after adding any specified regularization penalty to the loss (if training).
        """
        batch = nn_util.move_to_device(batch, self.model_engine.device)
        output_dict = self.model_engine(**batch)

        if for_training:
            try:
                assert "loss" in output_dict
                regularization_penalty = self.model.get_regularization_penalty()

                if regularization_penalty is not None:
                    output_dict["reg_loss"] = regularization_penalty
                    output_dict["loss"] += regularization_penalty

            except AssertionError:
                if for_training:
                    raise RuntimeError(
                        "The model you are trying to optimize does not contain a"
                        " 'loss' key in the output of model.forward(inputs)."
                    )

        return output_dict

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        cpu_memory_usage = []
        for worker, memory in common_util.peak_cpu_memory().items():
            cpu_memory_usage.append((worker, memory))
            logger.info(f"Worker {worker} memory usage: {common_util.format_size(memory)}")
        gpu_memory_usage = []
        for gpu, memory in common_util.peak_gpu_memory().items():
            gpu_memory_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage: {common_util.format_size(memory)}")

        regularization_penalty = self.model.get_regularization_penalty()

        train_loss = 0.0
        batch_loss = 0.0
        train_reg_loss = None if regularization_penalty is None else 0.0
        batch_reg_loss = None if regularization_penalty is None else 0.0

        # Set the model to "train" mode.
        self.model_engine.train()

        # Get tqdm for the training batches
        batch_generator = iter(self.data_loader)

        logger.info("Training")

        num_training_batches: Union[int, float]
        len_data_loader = len(self.data_loader)
        num_training_batches = len_data_loader

        # Having multiple tqdm bars in case of distributed training will be a mess. Hence only the master's
        # progress is shown
        if self._master:
            batch_generator_tqdm = Tqdm.tqdm(batch_generator, total=num_training_batches)
        else:
            batch_generator_tqdm = batch_generator

        self._last_log = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        for batch in batch_generator_tqdm:
            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            batch_outputs = self.batch_outputs(batch, for_training=True)

            loss = batch_outputs.get("loss")
            reg_loss = batch_outputs.get("reg_loss")
            if torch.isnan(loss):
                raise ValueError("nan loss encountered")

            batch_loss = 0 if loss is None else loss.item()
            train_loss += batch_loss
            if reg_loss is not None:
                batch_reg_loss = reg_loss.item()
                train_reg_loss += batch_reg_loss  # type: ignore

            self.model_engine.backward(loss)
            self.model_engine.step()

            param_updates = None
            if self._tensorboard.should_log_histograms_this_batch() and self._master:
                # Get the magnitude of parameter updates for logging.  We need to do some
                # computation before and after the optimizer step, and it's expensive because of
                # GPU/CPU copies (necessary for large models, and for shipping to tensorboard), so
                # we don't do this every batch, only when it's requested.
                param_updates = {
                    name: param.detach().cpu().clone()
                    for name, param in self.model.named_parameters()
                }

                for name, param in self.model.named_parameters():
                    param_updates[name].sub_(param.detach().cpu())

            # Update moving averages
            if self._moving_average is not None:
                self._moving_average.apply(batch_num_total)

            # Update the description with the latest metrics
            metrics = training_util.get_metrics(
                self.model,
                train_loss,
                train_reg_loss,
                batch_loss,
                batch_reg_loss,
                batches_this_epoch,
                world_size=self._world_size,
                cuda_device=self.cuda_device,
            )

            if self._master:
                # Updating tqdm only for the master as the trainers wouldn't have one
                description = training_util.description_from_metrics(metrics)
                batch_generator_tqdm.set_description(description, refresh=False)
                self._tensorboard.log_batch(
                    self.model,
                    self.optimizer,
                    0.0,
                    metrics,
                    batch,
                    param_updates,
                )

                if self._checkpointer is not None:
                    self._checkpointer.maybe_save_checkpoint(self, epoch, batches_this_epoch)

            for callback in self._batch_callbacks:
                callback(
                    self,
                    batch,
                    [batch_outputs],
                    metrics,
                    epoch,
                    batches_this_epoch,
                    is_training=True,
                    is_master=self._master,
                )

        metrics = training_util.get_metrics(
            self.model,
            train_loss,
            train_reg_loss,
            batch_loss=None,
            batch_reg_loss=None,
            num_batches=batches_this_epoch,
            reset=True,
            world_size=self._world_size,
            cuda_device=self.cuda_device,
        )

        for (worker, memory) in cpu_memory_usage:
            metrics["worker_" + str(worker) + "_memory_MB"] = memory / (1024 * 1024)
        for (gpu_num, memory) in gpu_memory_usage:
            metrics["gpu_" + str(gpu_num) + "_memory_MB"] = memory / (1024 * 1024)
        return metrics

    def _try_train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """
        try:
            epoch_counter = self._restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise ConfigurationError(
                "Could not recover training from the checkpoint.  Did you mean to output to "
                "a different serialization directory or delete the existing serialization "
                "directory?"
            )

        logger.info("Beginning training.")

        val_metrics: Dict[str, float] = {}
        this_epoch_val_metric: float = 0.0
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()

        metrics["best_epoch"] = self._metric_tracker.best_epoch
        for key, value in self._metric_tracker.best_epoch_metrics.items():
            metrics["best_validation_" + key] = value

        for callback in self._epoch_callbacks:
            callback(self, metrics={}, epoch=-1, is_master=self._master)

        for epoch in range(epoch_counter, self._num_epochs):
            epoch_start_time = time.time()
            train_metrics = self._train_epoch(epoch)

            # get peak of memory usage
            for key, value in train_metrics.items():
                if key.startswith("gpu_") and key.endswith("_memory_MB"):
                    metrics["peak_" + key] = max(metrics.get("peak_" + key, 0), value)
                elif key.startswith("worker_") and key.endswith("_memory_MB"):
                    metrics["peak_" + key] = max(metrics.get("peak_" + key, 0), value)

            if self._validation_data_loader is not None:
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it.
                    val_loss, val_reg_loss, num_batches = self._validation_loss(epoch)

                    # It is safe again to wait till the validation is done. This is
                    # important to get the metrics right.
                    if self._distributed:
                        dist.barrier()

                    val_metrics = training_util.get_metrics(
                        self.model,
                        val_loss,
                        val_reg_loss,
                        batch_loss=None,
                        batch_reg_loss=None,
                        num_batches=num_batches,
                        reset=True,
                        world_size=self._world_size,
                        cuda_device=self.cuda_device,
                    )

                    # Check validation metric for early stopping
                    this_epoch_val_metric = val_metrics[self._validation_metric]
                    self._metric_tracker.add_metric(this_epoch_val_metric)

                    if self._metric_tracker.should_stop_early():
                        logger.info("Ran out of patience.  Stopping training.")
                        break

            if self._master:
                self._tensorboard.log_metrics(
                    train_metrics, val_metrics=val_metrics, log_to_console=True, epoch=epoch + 1
                )  # +1 because tensorboard doesn't like 0

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
            metrics["training_start_epoch"] = epoch_counter
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value

            if self._metric_tracker.is_best_so_far():
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                metrics["best_epoch"] = epoch
                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value

                self._metric_tracker.best_epoch_metrics = val_metrics

            if self._serialization_dir and self._master:
                common_util.dump_metrics(
                    os.path.join(self._serialization_dir, f"metrics_epoch_{epoch}.json"), metrics
                )

            # deepspeed checkpointing handles master / dist.barrier calls
            if self._checkpointer is not None:
                self._checkpointer.save_checkpoint(
                    epoch, self, is_best_so_far=self._metric_tracker.is_best_so_far()
                )

            for callback in self._epoch_callbacks:
                callback(self, metrics=metrics, epoch=epoch, is_master=self._master)

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            if epoch < self._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * (
                    (self._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1
                )
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            epochs_trained += 1

        # make sure pending events are flushed to disk and files are closed properly
        self._tensorboard.close()

        # Load the best model state before returning
        best_model_state = (
            None if self._checkpointer is None else self._checkpointer.best_model_state()
        )
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return metrics

    def _restore_checkpoint(self) -> int:
        """
        Restores the model and training state from the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        ` model.load_state_dict(torch.load("/path/to/model/weights.th"))`
        If `self._serialization_dir` does not exist or does not contain any checkpointed weights,
        this function will do nothing and return 0.
        # Returns
        epoch: `int`
            The epoch at which to resume training, which should be one after the epoch
            in the saved training state.
        """
        if self._checkpointer is None:
            return 0

        self._checkpointer: DeepspeedCheckpointer
        checkpoint_id, model_state, training_state = self._checkpointer.restore_checkpoint()

        if not training_state:
            # No checkpoint to restore, start at 0
            return 0

        self.model.load_state_dict(model_state)
        self.model_engine.load_checkpoint(self._serialization_dir, checkpoint_id)

        # Currently the `training_state` contains a serialized `MetricTracker`.
        if "metric_tracker" in training_state:
            self._metric_tracker.load_state_dict(training_state["metric_tracker"])
        # It used to be the case that we tracked `val_metric_per_epoch`.
        elif "val_metric_per_epoch" in training_state:
            self._metric_tracker.clear()
            self._metric_tracker.add_metrics(training_state["val_metric_per_epoch"])
        # And before that we didn't track anything.
        else:
            self._metric_tracker.clear()

        if isinstance(training_state["epoch"], int):
            epoch_to_return = training_state["epoch"] + 1
        else:
            epoch_to_return = int(training_state["epoch"].split(".")[0]) + 1

        # For older checkpoints with batch_num_total missing, default to old behavior where
        # it is unchanged.
        batch_num_total = training_state.get("batch_num_total")
        if batch_num_total is not None:
            self._batch_num_total = batch_num_total

        return epoch_to_return

    @classmethod
    @overrides
    def from_partial_objects(
        cls,
        model: Model,
        serialization_dir: str,
        data_loader: DataLoader,
        deepspeed_config: DeepspeedConfig,
        validation_data_loader: DataLoader = None,
        local_rank: int = 0,
        patience: int = None,
        validation_metric: str = "-loss",
        num_epochs: int = 20,
        cuda_device: Optional[Union[int, torch.device]] = None,
        distributed: bool = None,
        world_size: int = 1,
        num_gradient_accumulation_steps: int = 1,
        no_grad: List[str] = None,
        optimizer: Lazy[Optimizer] = Lazy(Optimizer.default),
        deepspeed_optimizer: Dict[str, Any] = None,
        deepspeed_args: Lazy[DeepspeedArgs] = Lazy(DeepspeedArgs),
        tensorboard_writer: Lazy[TensorboardWriter] = Lazy(TensorboardWriter),
        moving_average: Lazy[MovingAverage] = None,
        checkpointer: Lazy[Checkpointer] = Lazy(DeepspeedCheckpointer),
        batch_callbacks: List[BatchCallback] = None,
        epoch_callbacks: List[EpochCallback] = None,
        end_callbacks: List[EpochCallback] = None,
        trainer_callbacks: List[TrainerCallback] = None,
    ) -> "DeepspeedTrainer":
        if no_grad:
            for name, parameter in model.named_parameters():
                if any(re.search(regex, name) for regex in no_grad):
                    parameter.requires_grad_(False)

        common_util.log_frozen_and_tunable_parameter_names(model)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        moving_average_ = (
            None if moving_average is None else moving_average.construct(parameters=parameters)
        )

        checkpointer_ = checkpointer.construct(serialization_dir=serialization_dir)
        tensorboard_writer_ = tensorboard_writer.construct(serialization_dir=serialization_dir)

        if deepspeed_config.optimizer:
            optim_ = None
        else:
            optim_ = optimizer.construct(model_parameters=parameters)

        deepspeed_args_ = deepspeed_args.construct(local_rank=local_rank) or DeepspeedArgs(
            local_rank=local_rank
        )

        if not hasattr(data_loader, 'batch_size'):
            raise ConfigurationError("Please specify your batch size in Deepspeed config if not using AllennlpDataLoader.")

        model_engine, ds_optimizer = _launch_deepspeed(
            model,
            optim_,
            deepspeed_config,
            deepspeed_args_,
            data_loader.batch_size, # type: ignore
            num_gradient_accumulation_steps,
        )

        return cls(
            model,
            data_loader,
            deepspeed_engine=model_engine,
            patience=patience,
            validation_metric=validation_metric,
            validation_data_loader=validation_data_loader,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            tensorboard_writer=tensorboard_writer_,
            checkpointer=checkpointer_,
            moving_average=moving_average_,
            batch_callbacks=batch_callbacks,
            epoch_callbacks=epoch_callbacks,
            end_callbacks=end_callbacks,
            trainer_callbacks=trainer_callbacks,
            distributed=False,
            local_rank=local_rank,
            world_size=world_size,
            num_gradient_accumulation_steps=num_gradient_accumulation_steps,
        )


def _launch_deepspeed(
    model: Model,
    optimizer: torch.optim.Optimizer,
    deepspeed_config: DeepspeedConfig,
    args: DeepspeedArgs,
    batch_size: int,
    gradient_accumulation_steps: int,
):
    if not (optimizer is None or deepspeed_config.optimizer is None):
        raise ConfigurationError(
            f"Cannot provide both optimizer and deepspeed_optimizer. {optimizer, deepspeed_config.to_dict()}"
        )

    config: Dict[str, Any] = dict(
        **{k: v for k, v in deepspeed_config.to_dict().items() if v is not None},
        train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    ds = DeepSpeedEngine(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
        dist_init_required=False,
        config_params=config,
    )
    if hasattr(ds, "timers"):

        def mute_log(*args, **kwargs):
            pass

        ds.timers.log = mute_log
    return ds, ds.optimizer
