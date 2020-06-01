import datetime
import logging
import math
import os
import re
import time
import traceback
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from allennlp.common.util import int_to_device

try:
    from apex import amp
except ImportError:
    amp = None
import torch
import torch.distributed as dist
import torch.optim.lr_scheduler
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_

from allennlp.common import Lazy, Registrable, Tqdm
from allennlp.common import util as common_util
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.data import DataLoader
from allennlp.data.dataloader import TensorDict
from allennlp.models.model import Model
from allennlp.nn import util as nn_util
from allennlp.training import util as training_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer
from allennlp.training.tensorboard_writer import TensorboardWriter

logger = logging.getLogger(__name__)


class Trainer(Registrable):
    """
    The base class for an AllenNLP trainer. It can do pretty much
    anything you want. Your subclass should implement `train`
    and also probably `from_params`.
    """

    default_implementation = "gradient_descent"

    def __init__(
        self,
        serialization_dir: str,
        cuda_device: Union[int, torch.device] = -1,
        distributed: bool = False,
        local_rank: int = 0,
        world_size: int = 1,
    ) -> None:

        check_for_gpu(cuda_device)
        self._serialization_dir = serialization_dir

        if isinstance(cuda_device, list):
            raise ConfigurationError(
                "In allennlp 1.0, the Trainer can only be assigned a single `cuda_device`. "
                "Instead, we use torch's DistributedDataParallel at the command level, meaning "
                "our Trainer always uses a single GPU per process."
            )

        if distributed and world_size <= 1:
            raise ConfigurationError(
                "Distributed training can be performed only with more than 1 device. Check "
                "`cuda_device` key in the experiment configuration."
            )

        self.cuda_device = int_to_device(cuda_device)

        self._distributed = distributed
        self._rank = local_rank
        self._master = self._rank == 0
        self._world_size = world_size

    def train(self) -> Dict[str, Any]:
        """
        Train a model and return the results.
        """
        raise NotImplementedError

    @contextmanager
    def get_checkpoint_state(self) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Returns a tuple of (model state, training state), where training state could have several
        internal components (e.g., for an, optimizer, learning rate scheduler, etc.).

        This is a context manager, and should be called as `with trainer.get_checkpoint_state() as
        state:`, so that the trainer has the opportunity to change and restore its internal state
        for checkpointing.  This is used, e.g., for moving averages of model weights.
        """
        raise NotImplementedError


class BatchCallback(Registrable):
    """
    An optional callback that you can pass to the `GradientDescentTrainer` that will be called at
    the end of every batch, during both training and validation.  The default implementation
    does nothing. You can implement your own callback and do whatever you want, such as saving
    predictions to disk or extra logging.
    """

    def __call__(
        self,
        trainer: "GradientDescentTrainer",
        batch_inputs: List[List[TensorDict]],
        batch_outputs: List[Dict[str, Any]],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_master: bool,
    ) -> None:
        pass


@BatchCallback.register("tensorboard-memory-usage")
class TensoboardBatchMemoryUsage(BatchCallback):
    """
    Logs the CPU and GPU memory usage to tensorboard on every batch.

    This is mainly used for debugging as it can cause a significant slowdown in training.
    """

    def __call__(
        self,
        trainer: "GradientDescentTrainer",
        batch_inputs: List[List[TensorDict]],
        batch_outputs: List[Dict[str, Any]],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_master: bool,
    ) -> None:
        if is_master:
            trainer._tensorboard.log_memory_usage()


BatchCallback.register("null")(BatchCallback)


class EpochCallback(Registrable):
    """
    An optional callback that you can pass to the `GradientDescentTrainer` that will be called at
    the end of every epoch (and before the start of training, with `epoch=-1`). The default
    implementation does nothing. You can implement your own callback and do whatever you want, such
    as additional modifications of the trainer's state in between epochs.
    """

    def __call__(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_master: bool,
    ) -> None:
        pass


EpochCallback.register("null")(EpochCallback)


@Trainer.register("gradient_descent", constructor="from_partial_objects")
class GradientDescentTrainer(Trainer):
    """
    A trainer for doing supervised learning with gradient descent. It just takes a labeled dataset
    and a `DataLoader`, and uses the supplied `Optimizer` to learn the weights for your model over
    some fixed number of epochs. You can also pass in a validation dataloader and enable early
    stopping. There are many other bells and whistles as well.

    Registered as a `Trainer` with the name "gradient_descent" (and is also the default `Trainer`).
    The constructor that is registered is `from_partial_objects` - see the arguments to that
    function for the exact keys that should be used, if you are using a configuration file.  They
    largely match the arguments to `__init__`, and we don't repeat their docstrings in
    `from_partial_objects`.

    [0]: https://tinyurl.com/y5mv44fw
    [1]: https://nvidia.github.io/apex/amp.html#opt-levels-and-properties

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
        A pytorch `DataLoader` containing your `Dataset`, yielding padded indexed batches.

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "trainer", it gets constructed separately.

    patience : `Optional[int] > 0`, optional (default=`None`)
        Number of epochs to be patient before early stopping: the training is stopped
        after `patience` epochs with no improvement. If given, it must be `> 0`.
        If None, early stopping is disabled.

    validation_metric : `str`, optional (default=`"loss"`)
        Validation metric to measure for whether to stop training using patience
        and whether to serialize an `is_best` model each epoch. The metric name
        must be prepended with either "+" or "-", which specifies whether the metric
        is an increasing or decreasing function.

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

    cuda_device : `int`, optional (default = `-1`)
        An integer specifying the CUDA device(s) to use for this process. If -1, the CPU is used.
        Data parallelism is controlled at the allennlp train level, so each trainer will have a single
        GPU.

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

    tensorboard_writer : `TensorboardWriter`, optional
        If this is not provided, we will construct a `TensorboardWriter` with default
        parameters and use that.

    moving_average : `MovingAverage`, optional, (default = `None`)
        If provided, we will maintain moving averages for all parameters. During training, we
        employ a shadow variable for each parameter, which maintains the moving average. During
        evaluation, we backup the original parameters and assign the moving averages to corresponding
        parameters. Be careful that when saving the checkpoint, we will save the moving averages of
        parameters. This is necessary because we want the saved model to perform as well as the validated
        model if we load it later. But this may cause problems if you restart the training from checkpoint.

    batch_callbacks : `List[BatchCallback]`, optional (default = `None`)
        A list of callbacks that will be called at the end of every batch, during both train and
        validation.

    epoch_callbacks : `List[EpochCallback]`, optional (default = `None`)
        A list of callbacks that will be called at the end of every epoch, and at the start of
        training (with epoch = -1).

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

    opt_level : `str`, optional, (default = `None`)
        Each opt_level establishes a set of properties that govern Amp’s implementation of pure or mixed
        precision training. Must be a choice of `"O0"`, `"O1"`, `"O2"`, or `"O3"`.
        See [the Apex documentation][1] for
        more details. If `None`, Amp is not used. Defaults to `None`.

    """

    def __init__(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        data_loader: torch.utils.data.DataLoader,
        patience: Optional[int] = None,
        validation_metric: str = "-loss",
        validation_data_loader: torch.utils.data.DataLoader = None,
        num_epochs: int = 20,
        serialization_dir: Optional[str] = None,
        checkpointer: Checkpointer = None,
        cuda_device: int = -1,
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
        learning_rate_scheduler: Optional[LearningRateScheduler] = None,
        momentum_scheduler: Optional[MomentumScheduler] = None,
        tensorboard_writer: TensorboardWriter = None,
        moving_average: Optional[MovingAverage] = None,
        batch_callbacks: List[BatchCallback] = None,
        epoch_callbacks: List[EpochCallback] = None,
        distributed: bool = False,
        local_rank: int = 0,
        world_size: int = 1,
        num_gradient_accumulation_steps: int = 1,
        opt_level: Optional[str] = None,
    ) -> None:
        super().__init__(serialization_dir, cuda_device, distributed, local_rank, world_size)

        # I am not calling move_to_gpu here, because if the model is
        # not already on the GPU then the optimizer is going to be wrong.
        self.model = model

        self.data_loader = data_loader
        self._validation_data_loader = validation_data_loader
        self.optimizer = optimizer

        if patience is None:  # no early stopping
            if validation_data_loader:
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

        self._learning_rate_scheduler = learning_rate_scheduler
        self._momentum_scheduler = momentum_scheduler
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

        # Enable automatic mixed precision training with NVIDIA Apex.
        self._opt_level = opt_level
        if self._opt_level is not None:
            if amp is None:
                raise ConfigurationError(
                    (
                        "Apex not installed but opt_level was provided. Please install NVIDIA's Apex to enable"
                        " automatic mixed precision (AMP) training. See: https://github.com/NVIDIA/apex."
                    )
                )

            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=self._opt_level
            )

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

    def rescale_gradients(self) -> Optional[float]:
        """
        Performs gradient rescaling. Is a no-op if gradient rescaling is not enabled.
        """
        if self._grad_norm:
            if self._opt_level is not None:
                # See: https://nvidia.github.io/apex/advanced.html#gradient-clipping
                parameters_to_clip = [
                    p for p in amp.master_params(self.optimizer) if p.grad is not None
                ]
            else:
                parameters_to_clip = [p for p in self.model.parameters() if p.grad is not None]
            return clip_grad_norm_(parameters_to_clip, self._grad_norm)
        else:
            return None

    def batch_outputs(self, batch: TensorDict, for_training: bool) -> Dict[str, torch.Tensor]:
        """
        Does a forward pass on the given batch and returns the output dictionary that the model
        returns, after adding any specified regularization penalty to the loss (if training).
        """
        batch = nn_util.move_to_device(batch, self.cuda_device)
        output_dict = self._pytorch_model(**batch)

        if for_training:
            try:
                regularization_penalty = self.model.get_regularization_penalty()
                loss = output_dict["loss"]

                # Handle model without regularization
                if regularization_penalty == 0.0:
                    regularization_penalty = loss.new_full(size=[], fill_value=0.0)

                output_dict["reg_loss"] = regularization_penalty
                output_dict["loss"] += regularization_penalty
            except KeyError:
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
        peak_cpu_usage = common_util.peak_memory_mb()
        logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
        gpu_usage = []
        for gpu, memory in common_util.gpu_memory_mb().items():
            gpu_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

        train_loss = 0.0
        train_reg_loss = 0.0
        # Set the model to "train" mode.
        self._pytorch_model.train()

        # Get tqdm for the training batches
        batch_generator = iter(self.data_loader)
        batch_group_generator = common_util.lazy_groups_of(
            batch_generator, self._num_gradient_accumulation_steps
        )

        logger.info("Training")

        num_training_batches = math.ceil(
            len(self.data_loader) / self._num_gradient_accumulation_steps
        )
        # Having multiple tqdm bars in case of distributed training will be a mess. Hence only the master's
        # progress is shown
        if self._master:
            batch_group_generator_tqdm = Tqdm.tqdm(
                batch_group_generator, total=num_training_batches
            )
        else:
            batch_group_generator_tqdm = batch_group_generator

        self._last_log = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        done_early = False
        for batch_group in batch_group_generator_tqdm:
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

            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            self.optimizer.zero_grad()

            batch_group_outputs = []
            for batch in batch_group:
                batch_outputs = self.batch_outputs(batch, for_training=True)
                batch_group_outputs.append(batch_outputs)
                loss = batch_outputs["loss"]
                reg_loss = batch_outputs["reg_loss"]
                if torch.isnan(loss):
                    raise ValueError("nan loss encountered")
                loss = loss / len(batch_group)
                reg_loss = reg_loss / len(batch_group)
                if self._opt_level is not None:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                train_loss += loss.item()
                train_reg_loss += reg_loss.item()

            batch_grad_norm = self.rescale_gradients()

            # This does nothing if batch_num_total is None or you are using a
            # scheduler which doesn't update per batch.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(batch_num_total)
            if self._momentum_scheduler:
                self._momentum_scheduler.step_batch(batch_num_total)

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
                self.optimizer.step()
                for name, param in self.model.named_parameters():
                    param_updates[name].sub_(param.detach().cpu())
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
                    batch_grad_norm,
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

        metrics = training_util.get_metrics(
            self.model,
            train_loss,
            train_reg_loss,
            batches_this_epoch,
            reset=True,
            world_size=self._world_size,
            cuda_device=self.cuda_device,
        )
        metrics["cpu_memory_MB"] = peak_cpu_usage
        for (gpu_num, memory) in gpu_usage:
            metrics["gpu_" + str(gpu_num) + "_memory_MB"] = memory
        return metrics

    def _validation_loss(self, epoch: int) -> Tuple[float, float, int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")

        self._pytorch_model.eval()

        # Replace parameter values with the shadow values from the moving averages.
        if self._moving_average is not None:
            self._moving_average.assign_average_value()

        if self._validation_data_loader is not None:
            validation_data_loader = self._validation_data_loader
        else:
            raise ConfigurationError(
                "Validation results cannot be calculated without a validation_data_loader"
            )

        val_generator_tqdm = Tqdm.tqdm(validation_data_loader)
        batches_this_epoch = 0
        val_loss = 0
        val_reg_loss = 0
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
                val_loss += loss.detach().cpu().numpy()
                if reg_loss is not None:
                    val_reg_loss += reg_loss.detach().cpu().numpy()

            # Update the description with the latest metrics
            val_metrics = training_util.get_metrics(
                self.model,
                val_loss,
                val_reg_loss,
                batches_this_epoch,
                world_size=self._world_size,
                cuda_device=self.cuda_device,
            )
            description = training_util.description_from_metrics(val_metrics)
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
            if "cpu_memory_MB" in train_metrics:
                metrics["peak_cpu_memory_MB"] = max(
                    metrics.get("peak_cpu_memory_MB", 0), train_metrics["cpu_memory_MB"]
                )
            for key, value in train_metrics.items():
                if key.startswith("gpu_"):
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
                        num_batches,
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

            # The Scheduler API is agnostic to whether your schedule requires a validation metric -
            # if it doesn't, the validation metric passed here is ignored.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step(this_epoch_val_metric)
            if self._momentum_scheduler:
                self._momentum_scheduler.step(this_epoch_val_metric)

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

        # If we have a learning rate or momentum scheduler, we should persist them too.
        if self._learning_rate_scheduler is not None:
            training_states["learning_rate_scheduler"] = self._learning_rate_scheduler.state_dict()
        if self._momentum_scheduler is not None:
            training_states["momentum_scheduler"] = self._momentum_scheduler.state_dict()
        # If model was trained with amp, we should persist the amp state.
        if self._opt_level is not None:
            training_states["amp"] = amp.state_dict()

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

        # The apex docs recommend calling amp.initialize before calling load_state_dict.
        if self._opt_level is not None and "amp" in training_state:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=self._opt_level
            )
            amp.load_state_dict(training_state["amp"])
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(training_state["optimizer"])
        if (
            self._learning_rate_scheduler is not None
            and "learning_rate_scheduler" in training_state
        ):
            self._learning_rate_scheduler.load_state_dict(training_state["learning_rate_scheduler"])
        if self._momentum_scheduler is not None and "momentum_scheduler" in training_state:
            self._momentum_scheduler.load_state_dict(training_state["momentum_scheduler"])
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
        cuda_device: int = -1,
        grad_norm: float = None,
        grad_clipping: float = None,
        distributed: bool = None,
        world_size: int = 1,
        num_gradient_accumulation_steps: int = 1,
        opt_level: Optional[str] = None,
        no_grad: List[str] = None,
        optimizer: Lazy[Optimizer] = None,
        learning_rate_scheduler: Lazy[LearningRateScheduler] = None,
        momentum_scheduler: Lazy[MomentumScheduler] = None,
        tensorboard_writer: Lazy[TensorboardWriter] = None,
        moving_average: Lazy[MovingAverage] = None,
        checkpointer: Lazy[Checkpointer] = None,
        batch_callbacks: List[BatchCallback] = None,
        epoch_callbacks: List[EpochCallback] = None,
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

        check_for_gpu(cuda_device)
        if cuda_device >= 0:
            # Moving model to GPU here so that the optimizer state gets constructed on
            # the right device.
            model = model.cuda(cuda_device)

        if no_grad:
            for name, parameter in model.named_parameters():
                if any(re.search(regex, name) for regex in no_grad):
                    parameter.requires_grad_(False)

        common_util.log_frozen_and_tunable_parameter_names(model)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer_ = optimizer.construct(model_parameters=parameters)
        if not optimizer_:
            optimizer_ = Optimizer.default(parameters)

        batches_per_epoch = len(data_loader)  # returns "1" instead of TypeError for _LazyInstances
        batches_per_epoch = math.ceil(batches_per_epoch / num_gradient_accumulation_steps)

        moving_average_ = moving_average.construct(parameters=parameters)
        learning_rate_scheduler_ = learning_rate_scheduler.construct(
            optimizer=optimizer_, num_epochs=num_epochs, num_steps_per_epoch=batches_per_epoch
        )
        momentum_scheduler_ = momentum_scheduler.construct(optimizer=optimizer_)

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
            learning_rate_scheduler=learning_rate_scheduler_,
            momentum_scheduler=momentum_scheduler_,
            tensorboard_writer=tensorboard_writer_,
            checkpointer=checkpointer_,
            moving_average=moving_average_,
            batch_callbacks=batch_callbacks,
            epoch_callbacks=epoch_callbacks,
            distributed=distributed,
            local_rank=local_rank,
            world_size=world_size,
            num_gradient_accumulation_steps=num_gradient_accumulation_steps,
            opt_level=opt_level,
        )
