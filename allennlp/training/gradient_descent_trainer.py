import datetime
import logging
import math
import os
import re
import time
import warnings
from typing import Optional, Union, List, Dict, Tuple, Any, Type

import torch
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist

from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common import util as common_util, Tqdm, Lazy
from allennlp.data.data_loaders.data_loader import DataLoader, TensorDict
from allennlp.models.model import Model
from allennlp.training.callbacks import ConsoleLoggerCallback
from allennlp.training.callbacks.confidence_checks import ConfidenceChecksCallback
from allennlp.training.callbacks.backward import MixedPrecisionBackwardCallback
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.momentum_schedulers.momentum_scheduler import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer import Trainer, TrainerCheckpoint
from allennlp.training.callbacks import TrainerCallback
from allennlp.training import util as training_util

logger = logging.getLogger(__name__)


@Trainer.register("gradient_descent", constructor="from_partial_objects")
class GradientDescentTrainer(Trainer):
    """
    A trainer for doing supervised learning with gradient descent. It just takes a labeled dataset
    and a `DataLoader`, and uses the supplied `Optimizer` to learn the weights for your model over
    some fixed number of epochs. You can also pass in a validation data_loader and enable early
    stopping. There are many other bells and whistles as well.

    Registered as a `Trainer` with the name "gradient_descent" (and is also the default `Trainer`).
    The constructor that is registered is [`from_partial_objects`](#from_partial_objects) -
    see the arguments to that function for the exact keys that should be used, if you are using
    a configuration file. They largely match the arguments to `__init__`, and we don't repeat their
    docstrings in `from_partial_objects`.

    [0]: https://tinyurl.com/y5mv44fw

    # Parameters

    model : `Model`, required.
        An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
        their `forward` method returns a dictionary with a "loss" key, containing a
        scalar tensor representing the loss function to be optimized.

        If you are training your model using GPUs, your model should already be
        on the correct device. (If you are using our `train` command this will be
        handled for you.)

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "trainer", it gets constructed separately.

    optimizer : `torch.nn.Optimizer`, required.
        An instance of a Pytorch Optimizer, instantiated with the parameters of the
        model to be optimized.

    data_loader : `DataLoader`, required.
        A `DataLoader` containing your `Dataset`, yielding padded indexed batches.

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "trainer", it gets constructed separately.

    patience : `Optional[int] > 0`, optional (default=`None`)
        Number of epochs to be patient before early stopping: the training is stopped
        after `patience` epochs with no improvement. If given, it must be `> 0`.
        If None, early stopping is disabled.

    validation_metric : `Union[str, List[str]]`, optional (default=`"-loss"`)
        Validation metric to measure for whether to stop training using patience
        and whether to serialize an `is_best` model each epoch. The metric name
        must be prepended with either "+" or "-", which specifies whether the metric
        is an increasing or decreasing function. If you specify more than one metric,
        the metrics will be summed to make the `is_best` decision.

    validation_data_loader : `DataLoader`, optional (default=`None`)
        A `DataLoader` to use for the validation set.  If `None`, then
        use the training `DataLoader` with the validation data.

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "trainer", it gets constructed separately.

    num_epochs : `int`, optional (default = `20`)
        Number of training epochs.

    serialization_dir : `str`, optional (default=`None`)
        Path to directory for saving and loading model files. Models will not be saved if
        this parameter is not passed.

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "trainer", it gets constructed separately.

    checkpointer : `Checkpointer`, optional (default=`None`)
        A `Checkpointer` is responsible for periodically saving model weights.  If none is given
        here, we will construct one with default parameters.

    cuda_device : `Optional[Union[int, torch.device]]`, optional (default = `None`)
        An integer or `torch.device` specifying the CUDA device to use for this process.
        If -1, the CPU is used. If `None` and you have a GPU available, that GPU will be used.

        !!! Note
            If you *don't* intend to use a GPU, but you have one available, you'll need
            to explicitly set `cuda_device=-1`.

        !!! Note
            If you intend to use a GPU, your model already needs to be on the correct device,
            which you can do with `model = model.cuda()`.

        !!! Note
            Data parallelism is controlled at the allennlp train level, so each trainer will have a single GPU.

    grad_norm : `float`, optional, (default = `None`).
        If provided, gradient norms will be rescaled to have a maximum of this value.

    grad_clipping : `float`, optional (default = `None`).
        If provided, gradients will be clipped `during the backward pass` to have an (absolute)
        maximum of this value.  If you are getting `NaNs` in your gradients during training
        that are not solved by using `grad_norm`, you may need this.

    learning_rate_scheduler : `LearningRateScheduler`, optional (default = `None`)
        If specified, the learning rate will be decayed with respect to
        this schedule at the end of each epoch (or batch, if the scheduler implements
        the `step_batch` method). If you use `torch.optim.lr_scheduler.ReduceLROnPlateau`,
        this will use the `validation_metric` provided to determine if learning has plateaued.
        To support updating the learning rate on every batch, this can optionally implement
        `step_batch(batch_num_total)` which updates the learning rate given the batch number.

    momentum_scheduler : `MomentumScheduler`, optional (default = `None`)
        If specified, the momentum will be updated at the end of each batch or epoch
        according to the schedule.

    moving_average : `MovingAverage`, optional, (default = `None`)
        If provided, we will maintain moving averages for all parameters. During training, we
        employ a shadow variable for each parameter, which maintains the moving average. During
        evaluation, we backup the original parameters and assign the moving averages to corresponding
        parameters. Be careful that when saving the checkpoint, we will save the moving averages of
        parameters. This is necessary because we want the saved model to perform as well as the validated
        model if we load it later. But this may cause problems if you restart the training from checkpoint.

    callbacks : `List[TrainerCallback]`, optional (default = `None`)
        A list of callbacks that can be called at certain events: e.g. each batch, epoch, and at the start
        and end of training, etc.

    distributed : `bool`, optional, (default = `False`)
        If set, PyTorch's `DistributedDataParallel` is used to train the model in multiple GPUs. This also
        requires `world_size` to be greater than 1.

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "trainer", it gets constructed separately (you need a top-level "distributed" key, next to
        the "trainer" entry, that specifies a list of "cuda_devices").

    local_rank : `int`, optional, (default = `0`)
        This is the unique identifier of the `Trainer` in a distributed process group. The GPU device id is
        used as the rank.

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "trainer", it gets constructed separately.

    world_size : `int`, (default = `1`)
        The number of `Trainer` workers participating in the distributed training.

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "trainer", it gets constructed separately.

    num_gradient_accumulation_steps : `int`, optional, (default = `1`)
        Gradients are accumulated for the given number of steps before doing an optimizer step. This can
        be useful to accommodate batches that are larger than the RAM size. Refer [Thomas Wolf's
        post][0] for details on Gradient Accumulation.

    use_amp : `bool`, optional, (default = `False`)
        If `True`, we'll train using [Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html).

    enable_default_callbacks : `bool`, optional (default = `True`)
        When `True`, the [`DEFAULT_CALLBACKS`](#default_callbacks) will be used in
        addition to any other callbacks listed in the `callbacks` parameter.
        When set to `False`, `DEFAULT_CALLBACKS` are not used.

    run_confidence_checks : `bool`, optional (default = `True`)
        Determines whether model confidence checks, such as
        [`NormalizationBiasVerification`](../../confidence_checks/normalization_bias_verification/),
        are run.

    run_sanity_checks : `bool`, optional (default = `True`)
        This parameter is deprecated. Please use `run_confidence_checks` instead.

    """

    def __init__(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        data_loader: DataLoader,
        patience: Optional[int] = None,
        validation_metric: Union[str, List[str]] = "-loss",
        validation_data_loader: DataLoader = None,
        num_epochs: int = 20,
        serialization_dir: Optional[str] = None,
        checkpointer: Checkpointer = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
        learning_rate_scheduler: Optional[LearningRateScheduler] = None,
        momentum_scheduler: Optional[MomentumScheduler] = None,
        moving_average: Optional[MovingAverage] = None,
        callbacks: List[TrainerCallback] = None,
        distributed: bool = False,
        local_rank: int = 0,
        world_size: int = 1,
        num_gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
        enable_default_callbacks: bool = True,
        run_confidence_checks: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            distributed=distributed,
            local_rank=local_rank,
            world_size=world_size,
        )

        if "run_sanity_checks" in kwargs:
            warnings.warn(
                "'run_sanity_checks' is deprecated, please use 'run_confidence_checks' instead.",
                DeprecationWarning,
            )
            run_confidence_checks = kwargs["run_sanity_checks"]

        # I am not calling move_to_gpu here, because if the model is
        # not already on the GPU then the optimizer is going to be wrong.
        self.model = model

        self.data_loader = data_loader
        self.data_loader.set_target_device(self.cuda_device)
        self._validation_data_loader = validation_data_loader
        if self._validation_data_loader is not None:
            self._validation_data_loader.set_target_device(self.cuda_device)
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
        self._metric_tracker = MetricTracker(validation_metric, patience)

        self._num_epochs = num_epochs

        self._checkpointer: Optional[Checkpointer] = checkpointer
        if checkpointer is None and serialization_dir is not None:
            self._checkpointer = Checkpointer(serialization_dir)

        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping

        self._learning_rate_scheduler = learning_rate_scheduler
        self._momentum_scheduler = momentum_scheduler
        self._moving_average = moving_average

        self._callbacks = callbacks or []
        default_callbacks = list(DEFAULT_CALLBACKS) if enable_default_callbacks else []

        if run_confidence_checks:
            default_callbacks.append(ConfidenceChecksCallback)
        for callback_cls in default_callbacks:
            for callback in self._callbacks:
                if callback.__class__ == callback_cls:
                    break
            else:
                self._callbacks.append(callback_cls(self._serialization_dir))

        self._num_gradient_accumulation_steps = num_gradient_accumulation_steps

        # Enable automatic mixed precision training.
        self._scaler: Optional[amp.GradScaler] = None
        self._use_amp = use_amp
        if self._use_amp:
            if self.cuda_device == torch.device("cpu"):
                raise ValueError("Using AMP requires a cuda device")
            self._scaler = amp.GradScaler()

        # Using `DistributedDataParallel`(ddp) brings in a quirk wrt AllenNLP's `Model` interface and its
        # usage. A `Model` object is wrapped by `ddp`, but assigning the wrapped model to `self.model`
        # will break the usages such as `Model.get_regularization_penalty`, `Model.get_metrics`, etc.
        #
        # Hence a reference to Pytorch's object is maintained in the case of distributed training and in the
        # normal case, reference to `Model` is retained. This reference is only used in
        # these places: `model.__call__`, `model.train` and `model.eval`.
        if self._distributed:
            self._pytorch_model = DistributedDataParallel(
                self.model,
                device_ids=None if self.cuda_device == torch.device("cpu") else [self.cuda_device],
                find_unused_parameters=True,
            )
        else:
            self._pytorch_model = self.model

        # training state management
        self._epochs_completed: int = 0
        self._start_after_epochs_completed: int = 0
        self._batches_in_epoch_completed: int = 0
        self._start_after_batches_in_epoch_completed: int = 0
        self._best_model_filename: Optional[str] = None

        # This is a kind of training state, but it is not serialized with the trainer state, because we can
        # re-create it with `epochs_completed` and `batches_in_epoch_completed`.
        self._total_batches_completed: int = 0

    def rescale_gradients(self) -> float:
        """
        Performs gradient rescaling. Is a no-op if gradient rescaling is not enabled.

        Returns the norm of the gradients.
        """
        parameters_to_clip = [p for p in self.model.parameters() if p.grad is not None]
        if self._grad_norm:
            if self._scaler is not None:
                # Need to first unscale gradients in order to clip as usual.
                self._scaler.unscale_(self.optimizer)
            return clip_grad_norm_(parameters_to_clip, self._grad_norm)
        else:
            return torch.norm(
                torch.stack([torch.norm(p.grad.detach()) for p in parameters_to_clip])
            )

    def batch_outputs(self, batch: TensorDict, for_training: bool) -> Dict[str, torch.Tensor]:
        """
        Does a forward pass on the given batch and returns the output dictionary that the model
        returns, after adding any specified regularization penalty to the loss (if training).
        """
        output_dict = self._pytorch_model(**batch)

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
        train_reg_loss = None if regularization_penalty is None else 0.0
        batch_reg_loss = None if regularization_penalty is None else 0.0

        # Set the model to "train" mode.
        self._pytorch_model.train()

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

        # Having multiple tqdm bars in case of distributed training will be a mess. Hence only the primary's
        # progress is shown
        if self._primary:
            batch_group_generator_tqdm = Tqdm.tqdm(
                batch_group_generator, total=num_training_batches
            )
        else:
            batch_group_generator_tqdm = batch_group_generator

        done_early = False
        for batch_group in batch_group_generator_tqdm:
            if done_early:
                break

            if self._epochs_completed < self._start_after_epochs_completed or (
                self._epochs_completed == self._start_after_epochs_completed
                and self._batches_in_epoch_completed < self._start_after_batches_in_epoch_completed
            ):
                self._batches_in_epoch_completed += 1
                self._total_batches_completed += 1
                continue

            self.optimizer.zero_grad()

            batch_loss = 0.0
            batch_group_outputs = []
            for batch in batch_group:
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
                            f"Worker {torch.distributed.get_rank()} finishing training early! "
                            "This implies that there is an imbalance in your training "
                            "data across the workers and that some amount of it will be "
                            "ignored. A small amount of this is fine, but a major imbalance "
                            "should be avoided. Note: This warning will appear unless your "
                            "data is perfectly balanced."
                        )
                        break

                with amp.autocast(self._use_amp):
                    batch_outputs = self.batch_outputs(batch, for_training=True)
                    batch_group_outputs.append(batch_outputs)
                    loss = batch_outputs["loss"]
                    reg_loss = batch_outputs.get("reg_loss")
                    if torch.isnan(loss):
                        raise ValueError("nan loss encountered")
                    loss = loss / len(batch_group)

                    batch_loss += loss.item()
                    if reg_loss is not None:
                        reg_loss = reg_loss / len(batch_group)
                        batch_reg_loss = reg_loss.item()
                        train_reg_loss += batch_reg_loss  # type: ignore

                backward_called = False
                for callback in self._callbacks:
                    backward_called |= callback.on_backward(self, batch_outputs, backward_called)
                if not backward_called:
                    if self._scaler is not None:
                        MixedPrecisionBackwardCallback(self._serialization_dir).on_backward(
                            self, batch_outputs, backward_called
                        )
                    else:
                        loss.backward()

            if len(batch_group_outputs) <= 0:
                continue

            train_loss += batch_loss

            batch_grad_norm = self.rescale_gradients()

            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(self._total_batches_completed + 1)
            if self._momentum_scheduler:
                self._momentum_scheduler.step_batch(self._total_batches_completed + 1)

            if self._scaler is not None:
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                self.optimizer.step()

            # Update moving averages
            if self._moving_average is not None:
                self._moving_average.apply(self._total_batches_completed + 1)

            self._batches_in_epoch_completed += 1
            self._total_batches_completed += 1

            # Update the description with the latest metrics
            metrics = training_util.get_metrics(
                self.model,
                train_loss,
                train_reg_loss,
                batch_loss,
                batch_reg_loss,
                self._batches_in_epoch_completed,
                world_size=self._world_size,
                cuda_device=self.cuda_device,
            )

            for callback in self._callbacks:
                callback.on_batch(
                    self,
                    batch_group,
                    batch_group_outputs,
                    metrics,
                    epoch,
                    self._batches_in_epoch_completed,
                    is_training=True,
                    is_primary=self._primary,
                    batch_grad_norm=batch_grad_norm,
                )

            if self._primary:
                # Updating tqdm only for the primary as the trainers wouldn't have one
                description = training_util.description_from_metrics(metrics)
                batch_group_generator_tqdm.set_description(description, refresh=False)

                if self._checkpointer is not None:
                    self._checkpointer.maybe_save_checkpoint(
                        self, self._epochs_completed, self._batches_in_epoch_completed
                    )

        if self._distributed and not done_early:
            logger.warning(
                f"Worker {torch.distributed.get_rank()} completed its entire epoch (training)."
            )
            # Indicate that we're done so that any workers that have remaining data stop the epoch early.
            done = torch.tensor(1, device=self.cuda_device)
            torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
            assert done.item()

        # Let all workers finish their epoch before computing
        # the final statistics for the epoch.
        if self._distributed:
            dist.barrier()

        if self._epochs_completed < self._start_after_epochs_completed or (
            self._epochs_completed == self._start_after_epochs_completed
            and self._batches_in_epoch_completed - 1 < self._start_after_batches_in_epoch_completed
        ):
            metrics = {}
        else:
            metrics = training_util.get_metrics(
                self.model,
                train_loss,
                train_reg_loss,
                batch_loss=None,
                batch_reg_loss=None,
                num_batches=self._batches_in_epoch_completed,
                reset=True,
                world_size=self._world_size,
                cuda_device=self.cuda_device,
            )

        for (worker, memory) in cpu_memory_usage:
            metrics["worker_" + str(worker) + "_memory_MB"] = memory / (1024 * 1024)
        for (gpu_num, memory) in gpu_memory_usage:
            metrics["gpu_" + str(gpu_num) + "_memory_MB"] = memory / (1024 * 1024)
        return metrics

    def _validation_loss(self, epoch: int) -> Tuple[float, Optional[float], int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")

        self._pytorch_model.eval()

        # Replace parameter values with the shadow values from the moving averages.
        if self._moving_average is not None:
            self._moving_average.assign_average_value()
        try:
            if self._validation_data_loader is not None:
                validation_data_loader = self._validation_data_loader
            else:
                raise ConfigurationError(
                    "Validation results cannot be calculated without a validation_data_loader"
                )

            regularization_penalty = self.model.get_regularization_penalty()

            # Having multiple tqdm bars in case of distributed training will be a mess. Hence only the primary's
            # progress is shown
            if self._primary:
                val_generator_tqdm = Tqdm.tqdm(validation_data_loader)
            else:
                val_generator_tqdm = validation_data_loader

            batches_this_epoch = 0
            val_loss = 0.0
            val_batch_loss = 0.0
            val_reg_loss = None if regularization_penalty is None else 0.0
            val_batch_reg_loss = None if regularization_penalty is None else 0.0
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
                        val_batch_loss = loss.item()
                        val_loss += val_batch_loss
                        if reg_loss is not None:
                            val_batch_reg_loss = reg_loss.item()
                            val_reg_loss += val_batch_reg_loss  # type: ignore

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
                if self._primary:
                    val_generator_tqdm.set_description(description, refresh=False)

                for callback in self._callbacks:
                    callback.on_batch(
                        self,
                        [batch],
                        [batch_outputs],
                        val_metrics,
                        epoch,
                        batches_this_epoch,
                        is_training=False,
                        is_primary=self._primary,
                    )

            if self._distributed and not done_early:
                logger.warning(
                    f"Worker {torch.distributed.get_rank()} completed its entire epoch (validation)."
                )
                # Indicate that we're done so that any workers that have remaining data stop validation early.
                done = torch.tensor(1, device=self.cuda_device)
                torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
                assert done.item()

            return val_loss, val_reg_loss, batches_this_epoch
        finally:
            # Now restore the original parameter values.
            if self._moving_average is not None:
                self._moving_average.restore()

    def train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """
        try:
            self._restore_checkpoint()
        except RuntimeError as e:
            configuration_error = ConfigurationError(
                "Could not recover training from the checkpoint. Did you mean to output to "
                "a different serialization directory or delete the existing serialization "
                "directory?"
            )
            configuration_error.__cause__ = e
            raise configuration_error

        # Callbacks get their `on_start` call even when we're starting from a checkpoint.
        for callback in self._callbacks:
            callback.on_start(self, is_primary=self._primary)

        # Set default values in case of failure
        epoch = None
        metrics = None

        try:
            metrics, epoch = self._try_train()
            return metrics
        finally:
            for callback in self._callbacks:
                callback.on_end(self, metrics=metrics, epoch=epoch, is_primary=self._primary)

    def _try_train(self) -> Tuple[Dict[str, Any], int]:
        training_util.enable_gradient_clipping(self.model, self._grad_clipping)

        logger.info("Beginning training.")

        val_metrics: Dict[str, float] = {}
        metrics: Dict[str, Any] = {}
        training_start_time = None

        metrics["best_epoch"] = self._metric_tracker.best_epoch
        for key, value in self._metric_tracker.best_epoch_metrics.items():
            metrics["best_validation_" + key] = value

        for epoch in range(self._num_epochs):
            epoch_start_time = time.time()
            train_metrics = self._train_epoch(epoch)

            if self._epochs_completed < self._start_after_epochs_completed:
                # We're still catching up with the checkpoint, so we do nothing.
                # Note that we have to call _train_epoch() even when we know the epoch is skipped. We have to
                # read from the data loader, because the data loader and dataset readers might use randomness,
                # and we have to make sure we consume exactly the same instances in exactly the same way every
                # time we train, even when starting from a checkpoint, so that we update the randomness
                # generators in the same way each time.
                self._epochs_completed += 1
                self._batches_in_epoch_completed = 0
                continue
            if training_start_time is None:
                training_start_time = epoch_start_time

            # get peak of memory usage
            for key, value in train_metrics.items():
                if key.startswith("gpu_") and key.endswith("_memory_MB"):
                    metrics["peak_" + key] = max(metrics.get("peak_" + key, 0), value)
                elif key.startswith("worker_") and key.endswith("_memory_MB"):
                    metrics["peak_" + key] = max(metrics.get("peak_" + key, 0), value)

            this_epoch_val_metric: float = 0.0
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
                    this_epoch_val_metric = self._metric_tracker.combined_score(val_metrics)
                    self._metric_tracker.add_metrics(val_metrics)

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
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

            if self._serialization_dir and self._primary:
                common_util.dump_metrics(
                    os.path.join(self._serialization_dir, f"metrics_epoch_{epoch}.json"),
                    metrics,
                )

            # The Scheduler API is agnostic to whether your schedule requires a validation metric -
            # if it doesn't, the validation metric passed here is ignored.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step(this_epoch_val_metric)
            if self._momentum_scheduler:
                self._momentum_scheduler.step(this_epoch_val_metric)
            for callback in self._callbacks:
                callback.on_epoch(self, metrics=metrics, epoch=epoch, is_primary=self._primary)

            self._epochs_completed += 1
            self._batches_in_epoch_completed = 0

            # The checkpointer saves state from the learning rate scheduler, momentum scheduler, moving
            # average, and callbacks, so we have to make sure those are updated before we save the
            # checkpoint here.
            if self._primary and self._checkpointer is not None:
                self._checkpointer.maybe_save_checkpoint(
                    self, self._epochs_completed, self._batches_in_epoch_completed
                )
            # Wait for the primary process to finish saving the checkpoint
            if self._distributed:
                dist.barrier()

            if self._primary and self._serialization_dir and self._metric_tracker.is_best_so_far():
                self._best_model_filename = os.path.join(self._serialization_dir, "best.th")
                if self._moving_average is None:
                    torch.save(self.model.state_dict(), self._best_model_filename)
                else:
                    self._moving_average.assign_average_value()
                    try:
                        torch.save(self.model.state_dict(), self._best_model_filename)
                    finally:
                        self._moving_average.restore()
            # Wait for the primary process to finish saving the best
            if self._distributed:
                dist.barrier()

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            if self._metric_tracker.should_stop_early():
                logger.info("Ran out of patience. Stopping training.")
                break

            if epoch < self._num_epochs - 1:
                time_per_epoch = training_elapsed_time / (
                    (epoch + 1) - self._start_after_epochs_completed
                )
                # Note: If the first non-skipped epoch is half skipped (because it was checkpointed half-way
                # through), then this estimate is going to be optimistic.
                estimated_time_remaining = (
                    time_per_epoch * self._num_epochs
                ) - training_elapsed_time
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)
        else:
            epoch = self._num_epochs - 1

        # Load the best model state before returning
        if self._best_model_filename is None or self._metric_tracker.is_best_so_far():
            self._finalize_model()
        else:
            # The model we're loading here has already been finalized.
            self.model.load_state_dict(torch.load(self._best_model_filename))

        return metrics, epoch

    def _finalize_model(self) -> None:
        """If we have a moving average, we have to finalize the model at the end of training."""
        if self._moving_average is not None:
            self._moving_average.assign_average_value()

    def get_checkpoint_state(self) -> TrainerCheckpoint:
        model_state = self.model.state_dict()

        # These are the training states we need to persist.
        training_states = {
            "version": 1,
            "metric_tracker": self._metric_tracker.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "callbacks": [cb.state_dict() for cb in self._callbacks],
            "epochs_completed": self._epochs_completed,
            "batches_in_epoch_completed": self._batches_in_epoch_completed,
            "best_model_filename": self._best_model_filename,
        }

        # If we have any of these optional objects, we should persist them too.
        if self._learning_rate_scheduler is not None:
            training_states["learning_rate_scheduler"] = self._learning_rate_scheduler.state_dict()
        if self._momentum_scheduler is not None:
            training_states["momentum_scheduler"] = self._momentum_scheduler.state_dict()
        if self._moving_average is not None:
            training_states["moving_average"] = self._moving_average.state_dict()

        return TrainerCheckpoint(model_state, training_states)

    def _restore_checkpoint(self) -> None:
        """
        Restores the model and training state from the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        `model.load_state_dict(torch.load("/path/to/model/weights.th"))`

        If `self._serialization_dir` does not exist or does not contain any checkpointed weights,
        this function will do nothing.
        """
        if self._checkpointer is None:
            return

        model_state, training_state = self._checkpointer.load_checkpoint()
        if len(model_state) <= 0 and len(training_state) <= 0:
            self._start_after_epochs_completed = 0
            self._start_after_batches_in_epoch_completed = 0
            self._best_model_filename = None
            return
        if training_state["version"] != 1:
            raise ValueError(
                f"This version of {self.__class__.__name__} only supports checkpoints of version 1. "
                f"Found version {training_state['version']}"
            )

        self.model.load_state_dict(model_state)
        self._metric_tracker.load_state_dict(training_state["metric_tracker"])
        self.optimizer.load_state_dict(training_state["optimizer"])

        for cb, state_dict in zip(self._callbacks, training_state["callbacks"]):
            cb.load_state_dict(state_dict)

        if self._learning_rate_scheduler is not None:
            self._learning_rate_scheduler.load_state_dict(training_state["learning_rate_scheduler"])
        if self._momentum_scheduler is not None:
            self._momentum_scheduler.load_state_dict(training_state["momentum_scheduler"])
        if self._moving_average is not None:
            self._moving_average.load_state_dict(training_state["moving_average"])

        self._start_after_epochs_completed = training_state["epochs_completed"]
        self._start_after_batches_in_epoch_completed = training_state["batches_in_epoch_completed"]
        self._best_model_filename = training_state["best_model_filename"]

    @classmethod
    def from_partial_objects(
        cls,
        model: Model,
        serialization_dir: str,
        data_loader: DataLoader,
        validation_data_loader: DataLoader = None,
        local_rank: int = 0,
        patience: int = None,
        validation_metric: Union[str, List[str]] = "-loss",
        num_epochs: int = 20,
        cuda_device: Optional[Union[int, torch.device]] = None,
        grad_norm: float = None,
        grad_clipping: float = None,
        distributed: bool = False,
        world_size: int = 1,
        num_gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
        no_grad: List[str] = None,
        optimizer: Lazy[Optimizer] = Lazy(Optimizer.default),
        learning_rate_scheduler: Lazy[LearningRateScheduler] = None,
        momentum_scheduler: Lazy[MomentumScheduler] = None,
        moving_average: Lazy[MovingAverage] = None,
        checkpointer: Lazy[Checkpointer] = Lazy(Checkpointer),
        callbacks: List[Lazy[TrainerCallback]] = None,
        enable_default_callbacks: bool = True,
        run_confidence_checks: bool = True,
        **kwargs,
    ) -> Trainer:
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

        common_util.log_frozen_and_tunable_parameter_names(model)

        batches_per_epoch: Optional[int]
        try:
            batches_per_epoch = len(data_loader)
            batches_per_epoch = math.ceil(batches_per_epoch / num_gradient_accumulation_steps)
        except TypeError:
            batches_per_epoch = None

        moving_average_ = (
            None if moving_average is None else moving_average.construct(parameters=parameters)
        )
        learning_rate_scheduler_ = (
            None
            if learning_rate_scheduler is None
            else learning_rate_scheduler.construct(
                optimizer=optimizer_, num_epochs=num_epochs, num_steps_per_epoch=batches_per_epoch
            )
        )
        momentum_scheduler_ = (
            None
            if momentum_scheduler is None
            else momentum_scheduler.construct(optimizer=optimizer_)
        )
        checkpointer_ = checkpointer.construct(serialization_dir=serialization_dir)

        callbacks_: List[TrainerCallback] = []
        for callback_ in callbacks or []:
            callbacks_.append(callback_.construct(serialization_dir=serialization_dir))

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
            learning_rate_scheduler=learning_rate_scheduler_,
            momentum_scheduler=momentum_scheduler_,
            checkpointer=checkpointer_,
            moving_average=moving_average_,
            callbacks=callbacks_,
            distributed=distributed,
            local_rank=local_rank,
            world_size=world_size,
            num_gradient_accumulation_steps=num_gradient_accumulation_steps,
            use_amp=use_amp,
            enable_default_callbacks=enable_default_callbacks,
            run_confidence_checks=run_confidence_checks,
            **kwargs,
        )

    def get_best_weights_path(self) -> Optional[str]:
        return self._best_model_filename


DEFAULT_CALLBACKS: Tuple[Type[TrainerCallback]] = (ConsoleLoggerCallback,)
"""
The default callbacks used by `GradientDescentTrainer`.
"""
