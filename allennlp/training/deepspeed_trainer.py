import logging
from deepspeed.utils import logger as ds_logger
ds_logger.setLevel(logging.WARNING)
ds_logger.propagate = False

import datetime
import logging
import os
import re
import math
import json
import tempfile
import time
import traceback
from copy import deepcopy
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from allennlp.common.util import int_to_device

import torch
import torch.distributed as dist
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_

import deepspeed

from allennlp.common import Lazy, Registrable, Tqdm, Params, FromParams
from allennlp.common import util as common_util
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.data import DataLoader
from allennlp.data.dataloader import TensorDict
from allennlp.models.model import Model
from allennlp.nn import util as nn_util
from allennlp.training import util as training_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer
from allennlp.training.tensorboard_writer import TensorboardWriter
from allennlp.training.trainer import Trainer, BatchCallback, EpochCallback

logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]

class DeepspeedConfig(FromParams):
    def __init__(
        self,
        optimizer: JsonDict,
        fp16: JsonDict = {'enabled': False},
        amp:  JsonDict = {'enabled': False},
        zero_optimization: Union[bool, Dict] = False,
        zero_allow_untested_optimizer: bool = True,
        wall_clock_breakdown: bool = False
    ):
        self.optimizer = optimizer
        self.fp16 = fp16
        self.amp = amp
        self.zero_optimization = zero_optimization
        self.zero_allow_untested_optimizer = zero_allow_untested_optimizer
        self.wall_clock_breakdown = wall_clock_breakdown

    @staticmethod
    def build_deepspeed_args(deepspeed_config_path: str, local_rank: int = 0):
        from argparse import ArgumentParser, Namespace
        parser = ArgumentParser()
        parser.add_argument('--local_rank', type=int, default=local_rank)
        parser = deepspeed.add_config_arguments(parser)

        args, _ = parser.parse_known_args()
        arg_dict = vars(args)

        arg_dict.update(dict(deepspeed_config=deepspeed_config_path, deepspeed=True, local_rank=local_rank))
        return Namespace(**arg_dict)

    @property
    def config(self):
        # return {
        #     'fp16': self.fp16,
        #     'amp': self.amp,
        #     'zero_optimization': self.zero_optimization,
        #     'zero_allow_untested_optimizer': self.zero_allow_untested_optimizer
        # }
        return vars(self)

    def _to_temp_file(self, serialization_dir, **kwargs):
        fd, path = tempfile.mkstemp(dir=serialization_dir)
        
        config = {**self.config, **kwargs}
        with os.fdopen(fd, 'w') as f:
            f.write(json.dumps(config))

        return path

    def launch(
        self,
        model: torch.nn.Module,
        optimizer: Union[str, torch.optim.Optimizer],
        local_rank: int, 
        serialization_dir: str,
        batch_size: int, 
        gradient_accumulation_steps: int,
        **kwargs
    ):
        path = self._to_temp_file(serialization_dir, train_batch_size=batch_size, gradient_accumulation_steps=gradient_accumulation_steps)
        ds = deepspeed.initialize(
            args=self.build_deepspeed_args(path, local_rank),
            model=model,
            model_parameters=model.parameters(),
            dist_init_required=False,
            **kwargs
        )

        os.remove(path)
        return ds

@Trainer.register("deepspeed", constructor="from_partial_objects")
class DeepspeedTrainer(Trainer):
    def __init__(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        data_loader: DataLoader,
        deepspeed_config: DeepspeedConfig,
        patience: Optional[int] = None,
        validation_metric: str = "-loss",
        validation_data_loader: DataLoader = None,
        num_epochs: int = 20,
        serialization_dir: Optional[str] = None,
        checkpointer: Checkpointer = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
        tensorboard_writer: TensorboardWriter = None,
        moving_average: Optional[MovingAverage] = None,
        batch_callbacks: List[BatchCallback] = None,
        epoch_callbacks: List[EpochCallback] = None,
        distributed: bool = False,
        local_rank: int = 0,
        world_size: int = 1,
        num_gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
    ) -> None:
        super().__init__(serialization_dir, cuda_device, distributed, local_rank, world_size)

        # I am not calling move_to_gpu here, because if the model is
        # not already on the GPU then the optimizer is going to be wrong.
        self.model = model

        self.data_loader = data_loader
        self._validation_data_loader = validation_data_loader
        self.optimizer = optimizer

        if patience is None:  # no early stopping
            if validation_data_loader is not None:
                logger.warning(
                    "You provided a validation dataset but patience was set to None, "
                    "meaning that early stopping is disabled"
                )
        elif (not isinstance(patience, int)) or patience <= 0:
            raise ConfigurationError(
                '{} is an invalid value for "patience": it must be a positive integer '
                "or None (if you want to disable early stopping)".format(patience)
            )

        # For tracking is_best_so_far and should_stop_early
        self._metric_tracker = MetricTracker(patience, validation_metric)
        # Get rid of + or -
        self._validation_metric = validation_metric[1:]

        self._num_epochs = num_epochs

        if checkpointer is not None:
            self._checkpointer = checkpointer
        else:
            self._checkpointer = Checkpointer(serialization_dir)

        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping

        self._moving_average = moving_average
        self._batch_callbacks = batch_callbacks or []
        self._epoch_callbacks = epoch_callbacks or []

        # We keep the total batch number as an instance variable because it
        # is used inside a closure for the hook which logs activations in
        # `_enable_activation_logging`.
        self._batch_num_total = 0

        self._tensorboard = tensorboard_writer or TensorboardWriter(serialization_dir)
        self._tensorboard.get_batch_num_total = lambda: self._batch_num_total
        self._tensorboard.enable_activation_logging(self.model)

        self._last_log = 0.0  # time of last logging

        self._num_gradient_accumulation_steps = num_gradient_accumulation_steps

        # Enable automatic mixed precision training.
        self._scaler: Optional[amp.GradScaler] = None
        self._use_amp = use_amp
        if self._use_amp:
            if self.cuda_device == torch.device("cpu"):
                raise ValueError("Using AMP requires a cuda device")
            self._scaler = amp.GradScaler()

        self._pytorch_model = self.model
        
        self._ds_config = deepspeed_config
        self.model_engine, self.ds_optimizer, _, _ = self._ds_config.launch(
            self.model,
            None, # self.optimizer,
            local_rank,
            serialization_dir,
            self.data_loader.batch_size,
            num_gradient_accumulation_steps
        )

        def mute_log(*args, **kwargs):
            pass

        if hasattr(self.model_engine, 'timers'):
            self.model_engine.timers.log = mute_log

    def batch_outputs(self, batch: TensorDict, for_training: bool) -> Dict[str, torch.Tensor]:
        """
        Does a forward pass on the given batch and returns the output dictionary that the model
        returns, after adding any specified regularization penalty to the loss (if training).
        """
        # batch = nn_util.move_to_device(batch, self.cuda_device)
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
        for worker, memory in common_util.peak_memory_mb().items():
            cpu_memory_usage.append((worker, memory))
            logger.info(f"Worker {worker} memory usage MB: {memory}")
        gpu_memory_usage = []
        for gpu, memory in common_util.gpu_memory_mb().items():
            gpu_memory_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

        regularization_penalty = self.model.get_regularization_penalty()

        train_loss = 0.0
        batch_loss = 0.0

        if regularization_penalty is not None:
            train_reg_loss = 0.0
            batch_reg_loss = 0.0
        else:
            train_reg_loss = None
            batch_reg_loss = None
        # Set the model to "train" mode.
        self.model_engine.train()

        # Get tqdm for the training batches
        batch_generator = iter(self.data_loader)
        batch_group_generator = common_util.lazy_groups_of(
            batch_generator, self._num_gradient_accumulation_steps
        )

        logger.info("Training")

        num_training_batches: Union[int, float]
        try:
            len_data_loader = len(self.data_loader)
            num_training_batches = math.ceil(
                len_data_loader / self._num_gradient_accumulation_steps
            )
        except TypeError:
            num_training_batches = float("inf")

        # Having multiple tqdm bars in case of distributed training will be a mess. Hence only the master's
        # progress is shown
        batch_group_generator_tqdm = batch_group_generator
        if self._master:
            batch_group_generator_tqdm = Tqdm.tqdm(
                batch_group_generator, total=num_training_batches
            )

        self._last_log = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        done_early = False
        for batch_group in batch_group_generator_tqdm:
            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            self.optimizer.zero_grad()

            batch_group_outputs = []
            for batch in batch_group:
                with amp.autocast(self._use_amp):
                    batch_outputs = self.batch_outputs(batch, for_training=True)
                    batch_group_outputs.append(batch_outputs)
                    loss = batch_outputs.get("loss")
                    reg_loss = batch_outputs.get("reg_loss")
                    if torch.isnan(loss):
                        raise ValueError("nan loss encountered")
                    loss = loss / len(batch_group)

                    batch_loss = loss.item()
                    train_loss += batch_loss
                    if reg_loss is not None:
                        reg_loss = reg_loss / len(batch_group)
                        batch_reg_loss = reg_loss.item()
                        train_reg_loss += batch_reg_loss

                
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

                if self._scaler is not None:
                    self._scaler.step(self.optimizer)
                    self._scaler.update()
                else:
                    self.optimizer.step()

                for name, param in self.model.named_parameters():
                    param_updates[name].sub_(param.detach().cpu())
            else:
                if self._scaler is not None:
                    self._scaler.step(self.optimizer)
                    self._scaler.update()
                else:
                    self.optimizer.step()

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
                batch_group_generator_tqdm.set_description(description, refresh=False)
                self._tensorboard.log_batch(
                    self.model,
                    self.optimizer,
                    0., # batch_grad_norm,
                    metrics,
                    batch_group,
                    param_updates,
                )

                self._checkpointer.maybe_save_checkpoint(self, epoch, batches_this_epoch)
            
            for callback in self._batch_callbacks:
                callback(
                    self,
                    batch_group,
                    batch_group_outputs,
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
            metrics["worker_" + str(worker) + "_memory_MB"] = memory
        for (gpu_num, memory) in gpu_memory_usage:
            metrics["gpu_" + str(gpu_num) + "_memory_MB"] = memory
        return metrics

    def _validation_loss(self, epoch: int) -> Tuple[float, float, int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")

        self.model_engine.eval()

        # Replace parameter values with the shadow values from the moving averages.
        if self._moving_average is not None:
            self._moving_average.assign_average_value()

        if self._validation_data_loader is not None:
            validation_data_loader = self._validation_data_loader
        else:
            raise ConfigurationError(
                "Validation results cannot be calculated without a validation_data_loader"
            )

        regularization_penalty = self.model.get_regularization_penalty()

        # Having multiple tqdm bars in case of distributed training will be a mess. Hence only the master's
        # progress is shown
        if self._master:
            val_generator_tqdm = Tqdm.tqdm(validation_data_loader)
        else:
            val_generator_tqdm = validation_data_loader

        batches_this_epoch = 0
        val_loss = 0
        val_batch_loss = 0
        if regularization_penalty is not None:
            val_reg_loss = 0
            val_batch_reg_loss = 0
        else:
            val_reg_loss = None
            val_batch_reg_loss = None
        done_early = False
        for batch in val_generator_tqdm:
            if self._distributed:
                # Check whether the other workers have stopped already (due to differing amounts of
                # data in each). If so, we can't proceed because we would hang when we hit the
                # barrier implicit in Model.forward. We use a IntTensor instead a BoolTensor
                # here because NCCL process groups apparently don't support BoolTensor.
                done = torch.tensor(0, device=self.cuda_device)
                torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
                if done.item() > 0:
                    done_early = True
                    logger.warning(
                        f"Worker {torch.distributed.get_rank()} finishing validation early! "
                        "This implies that there is an imbalance in your validation "
                        "data across the workers and that some amount of it will be "
                        "ignored. A small amount of this is fine, but a major imbalance "
                        "should be avoided. Note: This warning will appear unless your "
                        "data is perfectly balanced."
                    )
                    break

            with amp.autocast(self._use_amp):
                batch_outputs = self.batch_outputs(batch, for_training=False)
                loss = batch_outputs.get("loss")
                reg_loss = batch_outputs.get("reg_loss")
                if loss is not None:
                    # You shouldn't necessarily have to compute a loss for validation, so we allow for
                    # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                    # currently only used as the divisor for the loss function, so we can safely only
                    # count those batches for which we actually have a loss.  If this variable ever
                    # gets used for something else, we might need to change things around a bit.
                    batches_this_epoch += 1
                    val_batch_loss = loss.detach().cpu().numpy()
                    val_loss += val_batch_loss
                    if reg_loss is not None:
                        val_batch_reg_loss = reg_loss.detach().cpu().numpy()
                        val_reg_loss += val_batch_reg_loss

            # Update the description with the latest metrics
            val_metrics = training_util.get_metrics(
                self.model,
                val_loss,
                val_reg_loss,
                val_batch_loss,
                val_batch_reg_loss,
                batches_this_epoch,
                world_size=self._world_size,
                cuda_device=self.cuda_device,
            )

            description = training_util.description_from_metrics(val_metrics)
            if self._master:
                val_generator_tqdm.set_description(description, refresh=False)

            for callback in self._batch_callbacks:
                callback(
                    self,
                    [batch],
                    [batch_outputs],
                    epoch,
                    batches_this_epoch,
                    is_training=False,
                    is_master=self._master,
                )

        if self._distributed and not done_early:
            logger.warning(
                f"Worker {torch.distributed.get_rank()} completed its entire epoch (validation)."
            )
            # Indicate that we're done so that any workers that have remaining data stop validation early.
            done = torch.tensor(1, device=self.cuda_device)
            torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
            assert done.item()

        # Now restore the original parameter values.
        if self._moving_average is not None:
            self._moving_average.restore()

        return val_loss, val_reg_loss, batches_this_epoch

    def train(self) -> Dict[str, Any]:
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

        training_util.enable_gradient_clipping(self.model, self._grad_clipping)

        logger.info("Beginning training.")

        val_metrics: Dict[str, float] = {}
        this_epoch_val_metric: float = None
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


            if self._master:
                self._checkpointer.save_checkpoint(
                    epoch, self, is_best_so_far=self._metric_tracker.is_best_so_far()
                )

            # Wait for the master to finish saving the checkpoint
            if self._distributed:
                dist.barrier()

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
        best_model_state = self._checkpointer.best_model_state()
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return metrics

    @contextmanager
    def get_checkpoint_state(self) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
        if self._moving_average is not None:
            # Assigning average value to model parameters.  The checkpointer will call
            # `restore_state_after_checkpointing` when it is done to put this back to what it was.
            self._moving_average.assign_average_value()

        model_state = self.model.state_dict()

        # These are the training states we need to persist.
        training_states = {
            "metric_tracker": self._metric_tracker.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "batch_num_total": self._batch_num_total,
        }

        try:
            yield model_state, training_states
        finally:
            if self._moving_average is not None:
                self._moving_average.restore()

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
        model_state, training_state = self._checkpointer.restore_checkpoint()

        if not training_state:
            # No checkpoint to restore, start at 0
            return 0

        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(training_state["optimizer"])
        training_util.move_optimizer_to_cuda(self.optimizer)

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
    def from_partial_objects(
        cls,
        model: Model,
        serialization_dir: str,
        data_loader: DataLoader,
        validation_data_loader: DataLoader = None,
        local_rank: int = 0,
        patience: int = None,
        validation_metric: str = "-loss",
        num_epochs: int = 20,
        cuda_device: Optional[Union[int, torch.device]] = None,
        grad_norm: float = None,
        grad_clipping: float = None,
        distributed: bool = None,
        world_size: int = 1,
        num_gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
        no_grad: List[str] = None,
        optimizer: Lazy[Optimizer] = None,
        tensorboard_writer: Lazy[TensorboardWriter] = None,
        moving_average: Lazy[MovingAverage] = None,
        checkpointer: Lazy[Checkpointer] = None,
        batch_callbacks: List[BatchCallback] = None,
        epoch_callbacks: List[EpochCallback] = None,
        deepspeed_config: DeepspeedConfig = None
    ) -> "Trainer":
        """
        This method exists so that we can have a documented method to construct this class using
        `FromParams`. If you are not using `FromParams` or config files, you can safely ignore this
        method.
        The reason we can't just use `__init__` with `FromParams` here is because there are
        sequential dependencies to this class's arguments.  Anything that has a `Lazy[]` type
        annotation needs something from one of the non-`Lazy` arguments.  The `Optimizer` needs to
        have the parameters from the `Model` before it's constructed, and the `Schedulers` need to
        have the `Optimizer`. Because of this, the typical way we construct things `FromParams`
        doesn't work, so we use `Lazy` to allow for constructing the objects sequentially.
        If you're not using `FromParams`, you can just construct these arguments in the right order
        yourself in your code and call the constructor directly.
        """
        if cuda_device is None:
            from torch import cuda

            if cuda.device_count() > 0:
                cuda_device = 0
            else:
                cuda_device = -1

        check_for_gpu(cuda_device)
        if cuda_device >= 0:
            # Moving model to GPU here so that the optimizer state gets constructed on
            # the right device.
            model = model.cuda(cuda_device)

        if no_grad:
            for name, parameter in model.named_parameters():
                if any(re.search(regex, name) for regex in no_grad):
                    parameter.requires_grad_(False)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer_ = optimizer.construct(model_parameters=parameters)
        if not optimizer_:
            optimizer_ = Optimizer.default(parameters)

        common_util.log_frozen_and_tunable_parameter_names(model)

        batches_per_epoch: Optional[int]
        try:
            batches_per_epoch = len(data_loader)
            batches_per_epoch = math.ceil(batches_per_epoch / num_gradient_accumulation_steps)
        except TypeError:
            batches_per_epoch = None

        moving_average_ = moving_average.construct(parameters=parameters)

        checkpointer_ = checkpointer.construct() or Checkpointer(serialization_dir)
        tensorboard_writer_ = tensorboard_writer.construct() or TensorboardWriter(serialization_dir)

        return cls(
            model,
            optimizer_,
            data_loader,
            patience=patience,
            validation_metric=validation_metric,
            validation_data_loader=validation_data_loader,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            tensorboard_writer=tensorboard_writer_,
            checkpointer=checkpointer_,
            moving_average=moving_average_,
            batch_callbacks=batch_callbacks,
            epoch_callbacks=epoch_callbacks,
            distributed=distributed,
            local_rank=local_rank,
            world_size=world_size,
            num_gradient_accumulation_steps=num_gradient_accumulation_steps,
            use_amp=use_amp,
            deepspeed_config=deepspeed_config   
        )