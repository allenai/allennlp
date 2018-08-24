"""
A :class:`~allennlp.training.trainer.Trainer` is responsible for training a
:class:`~allennlp.models.model.Model`.

Typically you might create a configuration file specifying the model and
training parameters and then use :mod:`~allennlp.commands.train`
rather than instantiating a ``Trainer`` yourself.
"""
# pylint: disable=too-many-lines

import logging
import os
import shutil
import time
import re
import datetime
import traceback
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any, Set

import torch
import torch.optim.lr_scheduler
from torch.nn.parallel import replicate, parallel_apply
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from tensorboardX import SummaryWriter

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import peak_memory_mb, gpu_memory_mb
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def is_sparse(tensor):
    return tensor.is_sparse


def sparse_clip_norm(parameters, max_norm, norm_type=2) -> float:
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Supports sparse gradients.

    Parameters
    ----------
    parameters : ``(Iterable[torch.Tensor])``
        An iterable of Tensors that will have gradients normalized.
    max_norm : ``float``
        The max norm of the gradients.
    norm_type : ``float``
        The type of the used p-norm. Can be ``'inf'`` for infinity norm.

    Returns
    -------
    Total norm of the parameters (viewed as a single vector).
    """
    # pylint: disable=invalid-name,protected-access
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            if is_sparse(p.grad):
                # need to coalesce the repeated indices before finding norm
                grad = p.grad.data.coalesce()
                param_norm = grad._values().norm(norm_type)
            else:
                param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if is_sparse(p.grad):
                p.grad.data._values().mul_(clip_coef)
            else:
                p.grad.data.mul_(clip_coef)
    return total_norm


def move_optimizer_to_cuda(optimizer):
    """
    Move the optimizer state to GPU, if necessary.
    After calling, any parameter specific state in the optimizer
    will be located on the same device as the parameter.
    """
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            if param.is_cuda:
                param_state = optimizer.state[param]
                for k in param_state.keys():
                    if isinstance(param_state[k], torch.Tensor):
                        param_state[k] = param_state[k].cuda(device=param.get_device())


class TensorboardWriter:
    """
    Wraps a pair of ``SummaryWriter`` instances but is a no-op if they're ``None``.
    Allows Tensorboard logging without always checking for Nones first.
    """
    def __init__(self, train_log: SummaryWriter = None, validation_log: SummaryWriter = None) -> None:
        self._train_log = train_log
        self._validation_log = validation_log

    @staticmethod
    def _item(value: Any):
        if hasattr(value, 'item'):
            val = value.item()
        else:
            val = value
        return val

    def add_train_scalar(self, name: str, value: float, global_step: int) -> None:
        # get the scalar
        if self._train_log is not None:
            self._train_log.add_scalar(name, self._item(value), global_step)

    def add_train_histogram(self, name: str, values: torch.Tensor, global_step: int) -> None:
        if self._train_log is not None:
            if isinstance(values, torch.Tensor):
                values_to_write = values.cpu().data.numpy().flatten()
                self._train_log.add_histogram(name, values_to_write, global_step)

    def add_validation_scalar(self, name: str, value: float, global_step: int) -> None:

        if self._validation_log is not None:
            self._validation_log.add_scalar(name, self._item(value), global_step)


def time_to_str(timestamp: int) -> str:
    """
    Convert seconds past Epoch to human readable string.
    """
    datetimestamp = datetime.datetime.fromtimestamp(timestamp)
    return '{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}'.format(
            datetimestamp.year, datetimestamp.month, datetimestamp.day,
            datetimestamp.hour, datetimestamp.minute, datetimestamp.second
    )


def str_to_time(time_str: str) -> datetime.datetime:
    """
    Convert human readable string to datetime.datetime.
    """
    pieces: Any = [int(piece) for piece in time_str.split('-')]
    return datetime.datetime(*pieces)


class Trainer:
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 iterator: DataIterator,
                 train_dataset: Iterable[Instance],
                 validation_dataset: Optional[Iterable[Instance]] = None,
                 patience: Optional[int] = None,
                 validation_metric: str = "-loss",
                 validation_iterator: DataIterator = None,
                 shuffle: bool = True,
                 num_epochs: int = 20,
                 serialization_dir: Optional[str] = None,
                 num_serialized_models_to_keep: int = 20,
                 keep_serialized_model_every_num_seconds: int = None,
                 model_save_interval: float = None,
                 cuda_device: Union[int, List] = -1,
                 grad_norm: Optional[float] = None,
                 grad_clipping: Optional[float] = None,
                 learning_rate_scheduler: Optional[LearningRateScheduler] = None,
                 summary_interval: int = 100,
                 histogram_interval: int = None) -> None:
        """
        Parameters
        ----------
        model : ``Model``, required.
            An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
            their ``forward`` method returns a dictionary with a "loss" key, containing a
            scalar tensor representing the loss function to be optimized.
        optimizer : ``torch.nn.Optimizer``, required.
            An instance of a Pytorch Optimizer, instantiated with the parameters of the
            model to be optimized.
        iterator : ``DataIterator``, required.
            A method for iterating over a ``Dataset``, yielding padded indexed batches.
        train_dataset : ``Dataset``, required.
            A ``Dataset`` to train on. The dataset should have already been indexed.
        validation_dataset : ``Dataset``, optional, (default = None).
            A ``Dataset`` to evaluate on. The dataset should have already been indexed.
        patience : Optional[int] > 0, optional (default=None)
            Number of epochs to be patient before early stopping: the training is stopped
            after ``patience`` epochs with no improvement. If given, it must be ``> 0``.
            If None, early stopping is disabled.
        validation_metric : str, optional (default="loss")
            Validation metric to measure for whether to stop training using patience
            and whether to serialize an ``is_best`` model each epoch. The metric name
            must be prepended with either "+" or "-", which specifies whether the metric
            is an increasing or decreasing function.
        validation_iterator : ``DataIterator``, optional (default=None)
            An iterator to use for the validation set.  If ``None``, then
            use the training `iterator`.
        shuffle: ``bool``, optional (default=True)
            Whether to shuffle the instances in the iterator or not.
        num_epochs : int, optional (default = 20)
            Number of training epochs.
        serialization_dir : str, optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        num_serialized_models_to_keep : ``int``, optional (default=20)
            Number of previous model checkpoints to retain.  Default is to keep 20 checkpoints.
            A value of None or -1 means all checkpoints will be kept.
        keep_serialized_model_every_num_seconds : ``int``, optional (default=None)
            If num_serialized_models_to_keep is not None, then occasionally it's useful to
            save models at a given interval in addition to the last num_serialized_models_to_keep.
            To do so, specify keep_serialized_model_every_num_seconds as the number of seconds
            between permanently saved checkpoints.  Note that this option is only used if
            num_serialized_models_to_keep is not None, otherwise all checkpoints are kept.
        model_save_interval : ``float``, optional (default=None)
            If provided, then serialize models every ``model_save_interval``
            seconds within single epochs.  In all cases, models are also saved
            at the end of every epoch if ``serialization_dir`` is provided.
        cuda_device : ``int``, optional (default = -1)
            An integer specifying the CUDA device to use. If -1, the CPU is used.
        grad_norm : ``float``, optional, (default = None).
            If provided, gradient norms will be rescaled to have a maximum of this value.
        grad_clipping : ``float``, optional (default = ``None``).
            If provided, gradients will be clipped `during the backward pass` to have an (absolute)
            maximum of this value.  If you are getting ``NaNs`` in your gradients during training
            that are not solved by using ``grad_norm``, you may need this.
        learning_rate_scheduler : ``PytorchLRScheduler``, optional, (default = None)
            A Pytorch learning rate scheduler. The learning rate will be decayed with respect to
            this schedule at the end of each epoch. If you use
            :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`, this will use the ``validation_metric``
            provided to determine if learning has plateaued.  To support updating the learning
            rate on every batch, this can optionally implement ``step_batch(batch_num_total)`` which
            updates the learning rate given the batch number.
        summary_interval: ``int``, optional, (default = 100)
            Number of batches between logging scalars to tensorboard
        histogram_interval : ``int``, optional, (default = ``None``)
            If not None, then log histograms to tensorboard every ``histogram_interval`` batches.
            When this parameter is specified, the following additional logging is enabled:
                * Histograms of model parameters
                * The ratio of parameter update norm to parameter norm
                * Histogram of layer activations
            We log histograms of the parameters returned by
            ``model.get_parameters_for_histogram_tensorboard_logging``.
            The layer activations are logged for any modules in the ``Model`` that have
            the attribute ``should_log_activations`` set to ``True``.  Logging
            histograms requires a number of GPU-CPU copies during training and is typically
            slow, so we recommend logging histograms relatively infrequently.
            Note: only Modules that return tensors, tuples of tensors or dicts
            with tensors as values currently support activation logging.
        """
        self._model = model
        self._iterator = iterator
        self._validation_iterator = validation_iterator
        self._shuffle = shuffle
        self._optimizer = optimizer
        self._train_data = train_dataset
        self._validation_data = validation_dataset

        if patience is None:  # no early stopping
            if validation_dataset:
                logger.warning('You provided a validation dataset but patience was set to None, '
                               'meaning that early stopping is disabled')
        elif (not isinstance(patience, int)) or patience <= 0:
            raise ConfigurationError('{} is an invalid value for "patience": it must be a positive integer '
                                     'or None (if you want to disable early stopping)'.format(patience))
        self._patience = patience
        self._num_epochs = num_epochs

        self._serialization_dir = serialization_dir
        self._num_serialized_models_to_keep = num_serialized_models_to_keep
        self._keep_serialized_model_every_num_seconds = keep_serialized_model_every_num_seconds
        self._serialized_paths: List[Any] = []
        self._last_permanent_saved_checkpoint_time = time.time()
        self._model_save_interval = model_save_interval

        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping
        self._learning_rate_scheduler = learning_rate_scheduler

        increase_or_decrease = validation_metric[0]
        if increase_or_decrease not in ["+", "-"]:
            raise ConfigurationError("Validation metrics must specify whether they should increase "
                                     "or decrease by pre-pending the metric name with a +/-.")
        self._validation_metric = validation_metric[1:]
        self._validation_metric_decreases = increase_or_decrease == "-"

        if not isinstance(cuda_device, int) and not isinstance(cuda_device, list):
            raise ConfigurationError("Expected an int or list for cuda_device, got {}".format(cuda_device))

        if isinstance(cuda_device, list):
            logger.warning(f"Multiple GPU support is experimental not recommended for use. "
                           "In some cases it may lead to incorrect results or undefined behavior.")
            self._multiple_gpu = True
            self._cuda_devices = cuda_device
            # data_parallel will take care of transfering to cuda devices,
            # so the iterator keeps data on CPU.
            self._iterator_device = -1
        else:
            self._multiple_gpu = False
            self._cuda_devices = [cuda_device]
            self._iterator_device = cuda_device

        if self._cuda_devices[0] != -1:
            self._model = self._model.cuda(self._cuda_devices[0])

        self._log_interval = 10  # seconds
        self._summary_interval = summary_interval
        self._histogram_interval = histogram_interval
        self._log_histograms_this_batch = False
        # We keep the total batch number as a class variable because it
        # is used inside a closure for the hook which logs activations in
        # ``_enable_activation_logging``.
        self._batch_num_total = 0

        self._last_log = 0.0  # time of last logging

        if serialization_dir is not None:
            train_log = SummaryWriter(os.path.join(serialization_dir, "log", "train"))
            validation_log = SummaryWriter(os.path.join(serialization_dir, "log", "validation"))
            self._tensorboard = TensorboardWriter(train_log, validation_log)
        else:
            self._tensorboard = TensorboardWriter()
        self._warned_tqdm_ignores_underscores = False

    def _enable_gradient_clipping(self) -> None:
        if self._grad_clipping is not None:
            # Pylint is unable to tell that we're in the case that _grad_clipping is not None...
            # pylint: disable=invalid-unary-operand-type
            clip_function = lambda grad: grad.clamp(-self._grad_clipping, self._grad_clipping)
            for parameter in self._model.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(clip_function)

    def _enable_activation_logging(self) -> None:
        """
        Log activations to tensorboard
        """
        if self._histogram_interval is not None:
            # To log activation histograms to the forward pass, we register
            # a hook on forward to capture the output tensors.
            # This uses a closure on self._log_histograms_this_batch to
            # determine whether to send the activations to tensorboard,
            # since we don't want them on every call.
            for _, module in self._model.named_modules():
                if not getattr(module, 'should_log_activations', False):
                    # skip it
                    continue

                def hook(module_, inputs, outputs):
                    # pylint: disable=unused-argument,cell-var-from-loop
                    log_prefix = 'activation_histogram/{0}'.format(module_.__class__)
                    if self._log_histograms_this_batch:
                        if isinstance(outputs, torch.Tensor):
                            log_name = log_prefix
                            self._tensorboard.add_train_histogram(log_name,
                                                                  outputs,
                                                                  self._batch_num_total)
                        elif isinstance(outputs, (list, tuple)):
                            for i, output in enumerate(outputs):
                                log_name = "{0}_{1}".format(log_prefix, i)
                                self._tensorboard.add_train_histogram(log_name,
                                                                      output,
                                                                      self._batch_num_total)
                        elif isinstance(outputs, dict):
                            for k, tensor in outputs.items():
                                log_name = "{0}_{1}".format(log_prefix, k)
                                self._tensorboard.add_train_histogram(log_name,
                                                                      tensor,
                                                                      self._batch_num_total)
                        else:
                            # skip it
                            pass

                module.register_forward_hook(hook)

    def _rescale_gradients(self) -> Optional[float]:
        """
        Performs gradient rescaling. Is a no-op if gradient rescaling is not enabled.
        """
        if self._grad_norm:
            parameters_to_clip = [p for p in self._model.parameters()
                                  if p.grad is not None]
            return sparse_clip_norm(parameters_to_clip, self._grad_norm)
        return None

    def _data_parallel(self, batch):
        """
        Do the forward pass using multiple GPUs.  This is a simplification
        of torch.nn.parallel.data_parallel to support the allennlp model
        interface.
        """
        inputs, module_kwargs = scatter_kwargs((), batch, self._cuda_devices, 0)
        used_device_ids = self._cuda_devices[:len(inputs)]
        replicas = replicate(self._model, used_device_ids)
        outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)

        # Only the 'loss' is needed.
        # a (num_gpu, ) tensor with loss on each GPU
        losses = gather([output['loss'].unsqueeze(0) for output in outputs], used_device_ids[0], 0)
        return {'loss': losses.mean()}

    def _batch_loss(self, batch: torch.Tensor, for_training: bool) -> torch.Tensor:
        """
        Does a forward pass on the given batch and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.
        """
        if self._multiple_gpu:
            output_dict = self._data_parallel(batch)
        else:
            output_dict = self._model(**batch)

        try:
            loss = output_dict["loss"]
            if for_training:
                loss += self._model.get_regularization_penalty()
        except KeyError:
            if for_training:
                raise RuntimeError("The model you are trying to optimize does not contain a"
                                   " 'loss' key in the output of model.forward(inputs).")
            loss = None

        return loss

    def _get_metrics(self, total_loss: float, num_batches: int, reset: bool = False) -> Dict[str, float]:
        """
        Gets the metrics but sets ``"loss"`` to
        the total loss divided by the ``num_batches`` so that
        the ``"loss"`` metric is "average loss per batch".
        """
        metrics = self._model.get_metrics(reset=reset)
        metrics["loss"] = float(total_loss / num_batches) if num_batches > 0 else 0.0
        return metrics

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        logger.info(f"Peak CPU memory usage MB: {peak_memory_mb()}")
        for gpu, memory in gpu_memory_mb().items():
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

        train_loss = 0.0
        # Set the model to "train" mode.
        self._model.train()

        # Get tqdm for the training batches
        train_generator = self._iterator(self._train_data,
                                         num_epochs=1,
                                         shuffle=self._shuffle,
                                         cuda_device=self._iterator_device)
        num_training_batches = self._iterator.get_num_batches(self._train_data)
        self._last_log = time.time()
        last_save_time = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        if self._histogram_interval is not None:
            histogram_parameters = set(self._model.get_parameters_for_histogram_tensorboard_logging())

        logger.info("Training")
        train_generator_tqdm = Tqdm.tqdm(train_generator,
                                         total=num_training_batches)
        for batch in train_generator_tqdm:
            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            self._log_histograms_this_batch = self._histogram_interval is not None and (
                    batch_num_total % self._histogram_interval == 0)

            self._optimizer.zero_grad()

            loss = self._batch_loss(batch, for_training=True)
            loss.backward()

            train_loss += loss.item()

            batch_grad_norm = self._rescale_gradients()

            # This does nothing if batch_num_total is None or you are using an
            # LRScheduler which doesn't update per batch.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(batch_num_total)

            if self._log_histograms_this_batch:
                # get the magnitude of parameter updates for logging
                # We need a copy of current parameters to compute magnitude of updates,
                # and copy them to CPU so large models won't go OOM on the GPU.
                param_updates = {name: param.detach().cpu().clone()
                                 for name, param in self._model.named_parameters()}
                self._optimizer.step()
                for name, param in self._model.named_parameters():
                    param_updates[name].sub_(param.detach().cpu())
                    update_norm = torch.norm(param_updates[name].view(-1, ))
                    param_norm = torch.norm(param.view(-1, )).cpu()
                    self._tensorboard.add_train_scalar("gradient_update/" + name,
                                                       update_norm / (param_norm + 1e-7),
                                                       batch_num_total)
            else:
                self._optimizer.step()

            # Update the description with the latest metrics
            metrics = self._get_metrics(train_loss, batches_this_epoch)
            description = self._description_from_metrics(metrics)

            train_generator_tqdm.set_description(description, refresh=False)

            # Log parameter values to Tensorboard
            if batch_num_total % self._summary_interval == 0:
                self._parameter_and_gradient_statistics_to_tensorboard(batch_num_total, batch_grad_norm)
                self._tensorboard.add_train_scalar("loss/loss_train", metrics["loss"], batch_num_total)
                self._metrics_to_tensorboard(batch_num_total,
                                             {"epoch_metrics/" + k: v for k, v in metrics.items()})

            if self._log_histograms_this_batch:
                self._histograms_to_tensorboard(batch_num_total, histogram_parameters)

            # Save model if needed.
            if self._model_save_interval is not None and (
                    time.time() - last_save_time > self._model_save_interval
            ):
                last_save_time = time.time()
                self._save_checkpoint(
                        '{0}.{1}'.format(epoch, time_to_str(int(last_save_time))), [], is_best=False
                )

        return self._get_metrics(train_loss, batches_this_epoch, reset=True)

    def _should_stop_early(self, metric_history: List[float]) -> bool:
        """
        uses patience and the validation metric to determine if training should stop early
        """
        if self._patience and self._patience < len(metric_history):
            # Pylint can't figure out that in this branch `self._patience` is an int.
            # pylint: disable=invalid-unary-operand-type

            # Is the best score in the past N epochs worse than or equal the best score overall?
            if self._validation_metric_decreases:
                return min(metric_history[-self._patience:]) >= min(metric_history[:-self._patience])
            else:
                return max(metric_history[-self._patience:]) <= max(metric_history[:-self._patience])

        return False

    def _parameter_and_gradient_statistics_to_tensorboard(self, # pylint: disable=invalid-name
                                                          epoch: int,
                                                          batch_grad_norm: float) -> None:
        """
        Send the mean and std of all parameters and gradients to tensorboard, as well
        as logging the average gradient norm.
        """
        # Log parameter values to Tensorboard
        for name, param in self._model.named_parameters():
            self._tensorboard.add_train_scalar("parameter_mean/" + name,
                                               param.data.mean(),
                                               epoch)
            self._tensorboard.add_train_scalar("parameter_std/" + name, param.data.std(), epoch)
            if param.grad is not None:
                if is_sparse(param.grad):
                    # pylint: disable=protected-access
                    grad_data = param.grad.data._values()
                else:
                    grad_data = param.grad.data

                # skip empty gradients
                if torch.prod(torch.tensor(grad_data.shape)).item() > 0: # pylint: disable=not-callable
                    self._tensorboard.add_train_scalar("gradient_mean/" + name,
                                                       grad_data.mean(),
                                                       epoch)
                    self._tensorboard.add_train_scalar("gradient_std/" + name,
                                                       grad_data.std(),
                                                       epoch)
                else:
                    # no gradient for a parameter with sparse gradients
                    logger.info("No gradient for %s, skipping tensorboard logging.", name)
        # norm of gradients
        if batch_grad_norm is not None:
            self._tensorboard.add_train_scalar("gradient_norm",
                                               batch_grad_norm,
                                               epoch)

    def _histograms_to_tensorboard(self, epoch: int, histogram_parameters: Set[str]) -> None:
        """
        Send histograms of parameters to tensorboard.
        """
        for name, param in self._model.named_parameters():
            if name in histogram_parameters:
                self._tensorboard.add_train_histogram("parameter_histogram/" + name,
                                                      param,
                                                      epoch)

    def _metrics_to_tensorboard(self,
                                epoch: int,
                                train_metrics: dict,
                                val_metrics: dict = None) -> None:
        """
        Sends all of the train metrics (and validation metrics, if provided) to tensorboard.
        """
        metric_names = set(train_metrics.keys())
        if val_metrics is not None:
            metric_names.update(val_metrics.keys())
        val_metrics = val_metrics or {}

        for name in metric_names:
            train_metric = train_metrics.get(name)
            if train_metric is not None:
                self._tensorboard.add_train_scalar(name, train_metric, epoch)
            val_metric = val_metrics.get(name)
            if val_metric is not None:
                self._tensorboard.add_validation_scalar(name, val_metric, epoch)

    def _metrics_to_console(self,  # pylint: disable=no-self-use
                            train_metrics: dict,
                            val_metrics: dict = None) -> None:
        """
        Logs all of the train metrics (and validation metrics, if provided) to the console.
        """
        val_metrics = val_metrics or {}
        dual_message_template = "%s |  %8.3f  |  %8.3f"
        no_val_message_template = "%s |  %8.3f  |  %8s"
        no_train_message_template = "%s |  %8s  |  %8.3f"
        header_template = "%s |  %-10s"

        metric_names = set(train_metrics.keys())
        if val_metrics:
            metric_names.update(val_metrics.keys())

        name_length = max([len(x) for x in metric_names])

        logger.info(header_template, "Training".rjust(name_length + 13), "Validation")
        for name in metric_names:
            train_metric = train_metrics.get(name)
            val_metric = val_metrics.get(name)

            if val_metric is not None and train_metric is not None:
                logger.info(dual_message_template, name.ljust(name_length), train_metric, val_metric)
            elif val_metric is not None:
                logger.info(no_train_message_template, name.ljust(name_length), "N/A", val_metric)
            elif train_metric is not None:
                logger.info(no_val_message_template, name.ljust(name_length), train_metric, "N/A")

    def _validation_loss(self) -> Tuple[float, int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")

        self._model.eval()

        if self._validation_iterator is not None:
            val_iterator = self._validation_iterator
        else:
            val_iterator = self._iterator

        val_generator = val_iterator(self._validation_data,
                                     num_epochs=1,
                                     shuffle=False,
                                     cuda_device=self._iterator_device)
        num_validation_batches = val_iterator.get_num_batches(self._validation_data)
        val_generator_tqdm = Tqdm.tqdm(val_generator,
                                       total=num_validation_batches)
        batches_this_epoch = 0
        val_loss = 0
        for batch in val_generator_tqdm:

            loss = self._batch_loss(batch, for_training=False)
            if loss is not None:
                # You shouldn't necessarily have to compute a loss for validation, so we allow for
                # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                # currently only used as the divisor for the loss function, so we can safely only
                # count those batches for which we actually have a loss.  If this variable ever
                # gets used for something else, we might need to change things around a bit.
                batches_this_epoch += 1
                val_loss += loss.detach().cpu().numpy()

            # Update the description with the latest metrics
            val_metrics = self._get_metrics(val_loss, batches_this_epoch)
            description = self._description_from_metrics(val_metrics)
            val_generator_tqdm.set_description(description, refresh=False)

        return val_loss, batches_this_epoch

    def train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """
        try:
            epoch_counter, validation_metric_per_epoch = self._restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise ConfigurationError("Could not recover training from the checkpoint.  Did you mean to output to "
                                     "a different serialization directory or delete the existing serialization "
                                     "directory?")

        self._enable_gradient_clipping()
        self._enable_activation_logging()

        logger.info("Beginning training.")

        train_metrics: Dict[str, float] = {}
        val_metrics: Dict[str, float] = {}
        best_epoch_val_metrics: Dict[str, float] = {}
        epochs_trained = 0
        training_start_time = time.time()
        for epoch in range(epoch_counter, self._num_epochs):
            epoch_start_time = time.time()
            train_metrics = self._train_epoch(epoch)

            if self._validation_data is not None:
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it.
                    val_loss, num_batches = self._validation_loss()
                    val_metrics = self._get_metrics(val_loss, num_batches, reset=True)

                    # Check validation metric for early stopping
                    this_epoch_val_metric = val_metrics[self._validation_metric]

                    # Check validation metric to see if it's the best so far
                    is_best_so_far = self._is_best_so_far(this_epoch_val_metric, validation_metric_per_epoch)
                    if is_best_so_far:
                        best_epoch_val_metrics = val_metrics.copy()
                    validation_metric_per_epoch.append(this_epoch_val_metric)
                    if self._should_stop_early(validation_metric_per_epoch):
                        logger.info("Ran out of patience.  Stopping training.")
                        break

            else:
                # No validation set, so just assume it's the best so far.
                is_best_so_far = True
                val_metrics = {}
                best_epoch_val_metrics = {}
                this_epoch_val_metric = None

            self._save_checkpoint(epoch, validation_metric_per_epoch, is_best=is_best_so_far)
            self._metrics_to_tensorboard(epoch, train_metrics, val_metrics=val_metrics)
            self._metrics_to_console(train_metrics, val_metrics)
            for index, param_group in enumerate(self._optimizer.param_groups):
                learning_rate = param_group.get("lr")
                if learning_rate is not None:
                    self._tensorboard.add_train_scalar(
                            f"learning_rate/param_group{index:d}", learning_rate, epoch)

            if self._learning_rate_scheduler:
                # The LRScheduler API is agnostic to whether your schedule requires a validation metric -
                # if it doesn't, the validation metric passed here is ignored.
                self._learning_rate_scheduler.step(this_epoch_val_metric, epoch)

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", time.strftime("%H:%M:%S", time.gmtime(epoch_elapsed_time)))

            if epoch < self._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * \
                    ((self._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1)
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            epochs_trained += 1

        training_elapsed_time = time.time() - training_start_time
        metrics = {
                "training_duration": time.strftime("%H:%M:%S", time.gmtime(training_elapsed_time)),
                "training_start_epoch": epoch_counter,
                "training_epochs": epochs_trained
        }
        for key, value in train_metrics.items():
            metrics["training_" + key] = value
        for key, value in val_metrics.items():
            metrics["validation_" + key] = value

        if validation_metric_per_epoch:
            # We may not have had validation data, so we need to hide this behind an if.
            if self._validation_metric_decreases:
                best_validation_metric = min(validation_metric_per_epoch)
            else:
                best_validation_metric = max(validation_metric_per_epoch)
            metrics.update({f"best_validation_{k}": v for k, v in best_epoch_val_metrics.items()})
            metrics['best_epoch'] = [i for i, value in enumerate(validation_metric_per_epoch)
                                     if value == best_validation_metric][-1]
        return metrics

    def _is_best_so_far(self,
                        this_epoch_val_metric: float,
                        validation_metric_per_epoch: List[float]):
        if not validation_metric_per_epoch:
            return True
        elif self._validation_metric_decreases:
            return this_epoch_val_metric < min(validation_metric_per_epoch)
        else:
            return this_epoch_val_metric > max(validation_metric_per_epoch)

    def _description_from_metrics(self, metrics: Dict[str, float]) -> str:
        if (not self._warned_tqdm_ignores_underscores and
                    any(metric_name.startswith("_") for metric_name in metrics)):
            logger.warning("Metrics with names beginning with \"_\" will "
                           "not be logged to the tqdm progress bar.")
            self._warned_tqdm_ignores_underscores = True
        return ', '.join(["%s: %.4f" % (name, value) for name, value in
                          metrics.items() if not name.startswith("_")]) + " ||"

    def _save_checkpoint(self,
                         epoch: Union[int, str],
                         val_metric_per_epoch: List[float],
                         is_best: Optional[bool] = None) -> None:
        """
        Saves a checkpoint of the model to self._serialization_dir.
        Is a no-op if self._serialization_dir is None.

        Parameters
        ----------
        epoch : Union[int, str], required.
            The epoch of training.  If the checkpoint is saved in the middle
            of an epoch, the parameter is a string with the epoch and timestamp.
        is_best: bool, optional (default = None)
            A flag which causes the model weights at the given epoch to
            be copied to a "best.th" file. The value of this flag should
            be based on some validation metric computed by your model.
        """
        if self._serialization_dir is not None:
            model_path = os.path.join(self._serialization_dir, "model_state_epoch_{}.th".format(epoch))
            model_state = self._model.state_dict()
            torch.save(model_state, model_path)

            training_state = {'epoch': epoch,
                              'val_metric_per_epoch': val_metric_per_epoch,
                              'optimizer': self._optimizer.state_dict(),
                              'batch_num_total': self._batch_num_total}
            training_path = os.path.join(self._serialization_dir,
                                         "training_state_epoch_{}.th".format(epoch))
            torch.save(training_state, training_path)
            if is_best:
                logger.info("Best validation performance so far. "
                            "Copying weights to '%s/best.th'.", self._serialization_dir)
                shutil.copyfile(model_path, os.path.join(self._serialization_dir, "best.th"))

            if self._num_serialized_models_to_keep and self._num_serialized_models_to_keep >= 0:
                self._serialized_paths.append([time.time(), model_path, training_path])
                if len(self._serialized_paths) > self._num_serialized_models_to_keep:
                    paths_to_remove = self._serialized_paths.pop(0)
                    # Check to see if we should keep this checkpoint, if it has been longer
                    # then self._keep_serialized_model_every_num_seconds since the last
                    # kept checkpoint.
                    remove_path = True
                    if self._keep_serialized_model_every_num_seconds is not None:
                        save_time = paths_to_remove[0]
                        time_since_checkpoint_kept = save_time - self._last_permanent_saved_checkpoint_time
                        if time_since_checkpoint_kept > self._keep_serialized_model_every_num_seconds:
                            # We want to keep this checkpoint.
                            remove_path = False
                            self._last_permanent_saved_checkpoint_time = save_time
                    if remove_path:
                        for fname in paths_to_remove[1:]:
                            os.remove(fname)

    def find_latest_checkpoint(self) -> Tuple[str, str]:
        """
        Return the location of the latest model and training state files.
        If there isn't a valid checkpoint then return None.
        """
        have_checkpoint = (self._serialization_dir is not None and
                           any("model_state_epoch_" in x for x in os.listdir(self._serialization_dir)))

        if not have_checkpoint:
            return None

        serialization_files = os.listdir(self._serialization_dir)
        model_checkpoints = [x for x in serialization_files if "model_state_epoch" in x]
        # Get the last checkpoint file.  Epochs are specified as either an
        # int (for end of epoch files) or with epoch and timestamp for
        # within epoch checkpoints, e.g. 5.2018-02-02-15-33-42
        found_epochs = [
                # pylint: disable=anomalous-backslash-in-string
                re.search("model_state_epoch_([0-9\.\-]+)\.th", x).group(1)
                for x in model_checkpoints
        ]
        int_epochs: Any = []
        for epoch in found_epochs:
            pieces = epoch.split('.')
            if len(pieces) == 1:
                # Just a single epoch without timestamp
                int_epochs.append([int(pieces[0]), 0])
            else:
                # has a timestamp
                int_epochs.append([int(pieces[0]), pieces[1]])
        last_epoch = sorted(int_epochs, reverse=True)[0]
        if last_epoch[1] == 0:
            epoch_to_load = str(last_epoch[0])
        else:
            epoch_to_load = '{0}.{1}'.format(last_epoch[0], last_epoch[1])

        model_path = os.path.join(self._serialization_dir,
                                  "model_state_epoch_{}.th".format(epoch_to_load))
        training_state_path = os.path.join(self._serialization_dir,
                                           "training_state_epoch_{}.th".format(epoch_to_load))

        return (model_path, training_state_path)

    def _restore_checkpoint(self) -> Tuple[int, List[float]]:
        """
        Restores a model from a serialization_dir to the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from  model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

        If ``self._serialization_dir`` does not exist or does not contain any checkpointed weights,
        this function will do nothing and return 0.

        Returns
        -------
        epoch: int
            The epoch at which to resume training, which should be one after the epoch
            in the saved training state.
        """
        latest_checkpoint = self.find_latest_checkpoint()

        if latest_checkpoint is None:
            # No checkpoint to restore, start at 0
            return 0, []

        model_path, training_state_path = latest_checkpoint

        # Load the parameters onto CPU, then transfer to GPU.
        # This avoids potential OOM on GPU for large models that
        # load parameters onto GPU then make a new GPU copy into the parameter
        # buffer. The GPU transfer happens implicitly in load_state_dict.
        model_state = torch.load(model_path, map_location=util.device_mapping(-1))
        training_state = torch.load(training_state_path, map_location=util.device_mapping(-1))
        self._model.load_state_dict(model_state)
        self._optimizer.load_state_dict(training_state["optimizer"])
        move_optimizer_to_cuda(self._optimizer)

        # We didn't used to save `validation_metric_per_epoch`, so we can't assume
        # that it's part of the trainer state. If it's not there, an empty list is all
        # we can do.
        if "val_metric_per_epoch" not in training_state:
            logger.warning("trainer state `val_metric_per_epoch` not found, using empty list")
            val_metric_per_epoch: List[float] = []
        else:
            val_metric_per_epoch = training_state["val_metric_per_epoch"]

        if isinstance(training_state["epoch"], int):
            epoch_to_return = training_state["epoch"] + 1
        else:
            epoch_to_return = int(training_state["epoch"].split('.')[0]) + 1

        # For older checkpoints with batch_num_total missing, default to old behavior where
        # it is unchanged.
        batch_num_total = training_state.get('batch_num_total')
        if batch_num_total is not None:
            self._batch_num_total = batch_num_total

        return epoch_to_return, val_metric_per_epoch

    # Requires custom from_params.
    @classmethod
    def from_params(cls,
                    model: Model,
                    serialization_dir: str,
                    iterator: DataIterator,
                    train_data: Iterable[Instance],
                    validation_data: Optional[Iterable[Instance]],
                    params: Params,
                    validation_iterator: DataIterator = None) -> 'Trainer':

        patience = params.pop_int("patience", None)
        validation_metric = params.pop("validation_metric", "-loss")
        shuffle = params.pop_bool("shuffle", True)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = params.pop_int("cuda_device", -1)
        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        lr_scheduler_params = params.pop("learning_rate_scheduler", None)

        if cuda_device >= 0:
            model = model.cuda(cuda_device)
        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))

        if lr_scheduler_params:
            scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
        else:
            scheduler = None

        num_serialized_models_to_keep = params.pop_int("num_serialized_models_to_keep", 20)
        keep_serialized_model_every_num_seconds = params.pop_int(
                "keep_serialized_model_every_num_seconds", None)
        model_save_interval = params.pop_float("model_save_interval", None)
        summary_interval = params.pop_int("summary_interval", 100)
        histogram_interval = params.pop_int("histogram_interval", None)

        params.assert_empty(cls.__name__)
        return Trainer(model, optimizer, iterator,
                       train_data, validation_data,
                       patience=patience,
                       validation_metric=validation_metric,
                       validation_iterator=validation_iterator,
                       shuffle=shuffle,
                       num_epochs=num_epochs,
                       serialization_dir=serialization_dir,
                       cuda_device=cuda_device,
                       grad_norm=grad_norm,
                       grad_clipping=grad_clipping,
                       learning_rate_scheduler=scheduler,
                       num_serialized_models_to_keep=num_serialized_models_to_keep,
                       keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds,
                       model_save_interval=model_save_interval,
                       summary_interval=summary_interval,
                       histogram_interval=histogram_interval)
