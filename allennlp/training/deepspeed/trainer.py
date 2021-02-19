import logging
import re
import time
from typing import Any, Dict, List, Optional, Union
from overrides import overrides

import torch
import torch.distributed as dist

from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.utils import logger as ds_logger

from allennlp.common import Lazy, Tqdm
from allennlp.common import util as common_util
from allennlp.common.checks import ConfigurationError
from allennlp.data import DataLoader, TensorDict
from allennlp.models.model import Model
from allennlp.nn import util as nn_util
from allennlp.training import util as training_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer

from allennlp.training.trainer import (
    Trainer,
    GradientDescentTrainer,
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
        moving_average: Optional[MovingAverage] = None,
        callbacks: List[TrainerCallback] = None,
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
            checkpointer=checkpointer,
            moving_average=moving_average,
            callbacks=callbacks,
            distributed=False,  # Avoid DDP init
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
        if self._primary:
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

            if self._primary:
                # Updating tqdm only for the master as the trainers wouldn't have one
                description = training_util.description_from_metrics(metrics)
                batch_generator_tqdm.set_description(description, refresh=False)

            if self._checkpointer is not None:
                self._checkpointer.maybe_save_checkpoint(self, epoch, batches_this_epoch)

            for callback in self._callbacks:
                callback.on_batch(
                    self,
                    batch,
                    batch_outputs,
                    metrics,
                    epoch,
                    batches_this_epoch,
                    is_training=True,
                    is_primary=self._primary,
                    batch_grad_norm=None,  # not yet implemented for DeepspeedTrainer
                )

        if self._distributed:
            dist.barrier()

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
        (
            checkpoint_id,
            model_state,
            training_state,
        ) = self._checkpointer.restore_checkpoint()

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
        moving_average: Lazy[MovingAverage] = None,
        checkpointer: Lazy[Checkpointer] = Lazy(DeepspeedCheckpointer),
        callbacks: List[Lazy[TrainerCallback]] = None,
        trainer_callbacks: List[Lazy[TrainerCallback]] = None,
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

        if deepspeed_config.optimizer:
            optim_ = None
        else:
            optim_ = optimizer.construct(model_parameters=parameters)

        deepspeed_args_ = deepspeed_args.construct(local_rank=local_rank) or DeepspeedArgs(
            local_rank=local_rank
        )

        if not hasattr(data_loader, "batch_size"):
            raise ConfigurationError(
                "Please specify your batch size in Deepspeed config if not using AllennlpDataLoader."
            )

        model_engine = DeepspeedTrainer._build_engine(
            model,
            optim_,
            deepspeed_config,
            deepspeed_args_,
            data_loader.batch_size,  # type: ignore
            num_gradient_accumulation_steps,
        )

        callbacks = callbacks or trainer_callbacks or []

        callbacks_: List[TrainerCallback] = []

        for callback in callbacks:
            callback_ = callback.construct(serialization_dir=serialization_dir)
            callbacks_.append(callback_)

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
            checkpointer=checkpointer_,
            moving_average=moving_average_,
            callbacks=callbacks_,
            distributed=False,
            local_rank=local_rank,
            world_size=world_size,
            num_gradient_accumulation_steps=num_gradient_accumulation_steps,
        )

    @staticmethod
    def _build_engine(
        model: Model,
        optimizer: torch.optim.Optimizer,
        deepspeed_config: DeepspeedConfig,
        args: DeepspeedArgs,
        batch_size: int,
        num_gradient_accumulation_steps: int,
    ):
        if not (optimizer is None or deepspeed_config.optimizer is None):
            raise ConfigurationError(
                f"Cannot provide both optimizer and deepspeed_optimizer. {optimizer, deepspeed_config.to_dict()}"
            )

        config: Dict[str, Any] = dict(
            **{k: v for k, v in deepspeed_config.to_dict().items() if v is not None},
            train_batch_size=batch_size,
            gradient_accumulation_steps=num_gradient_accumulation_steps,
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
        return ds
