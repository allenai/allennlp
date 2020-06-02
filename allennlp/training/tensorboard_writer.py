from typing import Any, Callable, Dict, List, Optional, Set
import logging
import os

from tensorboardX import SummaryWriter
import torch

from allennlp.common.util import peak_memory_mb, gpu_memory_mb
from allennlp.common.from_params import FromParams
from allennlp.data.dataloader import TensorDict
from allennlp.nn import util as nn_util
from allennlp.training.optimizers import Optimizer
from allennlp.training import util as training_util
from allennlp.models.model import Model

logger = logging.getLogger(__name__)


class TensorboardWriter(FromParams):
    """
    Class that handles Tensorboard (and other) logging.

    # Parameters

    serialization_dir : `str`, optional (default = `None`)
        If provided, this is where the Tensorboard logs will be written.

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "tensorboard_writer", it gets passed in separately.
    summary_interval : `int`, optional (default = `100`)
        Most statistics will be written out only every this many batches.
    histogram_interval : `int`, optional (default = `None`)
        If provided, activation histograms will be written out every this many batches.
        If None, activation histograms will not be written out.
        When this parameter is specified, the following additional logging is enabled:
            * Histograms of model parameters
            * The ratio of parameter update norm to parameter norm
            * Histogram of layer activations
        We log histograms of the parameters returned by
        `model.get_parameters_for_histogram_tensorboard_logging`.
        The layer activations are logged for any modules in the `Model` that have
        the attribute `should_log_activations` set to `True`.  Logging
        histograms requires a number of GPU-CPU copies during training and is typically
        slow, so we recommend logging histograms relatively infrequently.
        Note: only Modules that return tensors, tuples of tensors or dicts
        with tensors as values currently support activation logging.
    batch_size_interval : `int`, optional, (default = `None`)
        If defined, how often to log the average batch size.
    should_log_parameter_statistics : `bool`, optional (default = `True`)
        Whether to log parameter statistics (mean and standard deviation of parameters and
        gradients).
    should_log_learning_rate : `bool`, optional (default = `False`)
        Whether to log (parameter-specific) learning rate.
    get_batch_num_total : `Callable[[], int]`, optional (default = `None`)
        A thunk that returns the number of batches so far. Most likely this will
        be a closure around an instance variable in your `Trainer` class.  Because of circular
        dependencies in constructing this object and the `Trainer`, this is typically `None` when
        you construct the object, but it gets set inside the constructor of our `Trainer`.
    """

    def __init__(
        self,
        serialization_dir: Optional[str] = None,
        summary_interval: int = 100,
        histogram_interval: int = None,
        batch_size_interval: Optional[int] = None,
        should_log_parameter_statistics: bool = True,
        should_log_learning_rate: bool = False,
        get_batch_num_total: Callable[[], int] = None,
    ) -> None:
        if serialization_dir is not None:
            # Create log directories prior to creating SummaryWriter objects
            # in order to avoid race conditions during distributed training.
            train_ser_dir = os.path.join(serialization_dir, "log", "train")
            os.makedirs(train_ser_dir, exist_ok=True)
            self._train_log = SummaryWriter(train_ser_dir)
            val_ser_dir = os.path.join(serialization_dir, "log", "validation")
            os.makedirs(val_ser_dir, exist_ok=True)
            self._validation_log = SummaryWriter(val_ser_dir)
        else:
            self._train_log = self._validation_log = None

        self._summary_interval = summary_interval
        self._histogram_interval = histogram_interval
        self._batch_size_interval = batch_size_interval
        self._should_log_parameter_statistics = should_log_parameter_statistics
        self._should_log_learning_rate = should_log_learning_rate
        self.get_batch_num_total = get_batch_num_total

        self._cumulative_batch_group_size = 0
        self._batches_this_epoch = 0
        self._histogram_parameters: Set[str] = None

    @staticmethod
    def _item(value: Any):
        if hasattr(value, "item"):
            val = value.item()
        else:
            val = value
        return val

    def log_memory_usage(self):
        cpu_memory_usage = peak_memory_mb()
        self.add_train_scalar("memory_usage/cpu", cpu_memory_usage)
        for gpu, memory in gpu_memory_mb().items():
            self.add_train_scalar(f"memory_usage/gpu_{gpu}", memory)

    def log_batch(
        self,
        model: Model,
        optimizer: Optimizer,
        batch_grad_norm: Optional[float],
        metrics: Dict[str, float],
        batch_group: List[List[TensorDict]],
        param_updates: Optional[Dict[str, torch.Tensor]],
    ) -> None:
        if self.should_log_this_batch():
            self.log_parameter_and_gradient_statistics(model, batch_grad_norm)
            self.log_learning_rates(model, optimizer)

            self.add_train_scalar("loss/loss_train", metrics["loss"])
            self.log_metrics({"epoch_metrics/" + k: v for k, v in metrics.items()})

        if self.should_log_histograms_this_batch():
            self.log_histograms(model)
            self.log_gradient_updates(model, param_updates)

        if self._batch_size_interval:
            # We're assuming here that `log_batch` will get called every batch, and only every
            # batch.  This is true with our current usage of this code (version 1.0); if that
            # assumption becomes wrong, this code will break.
            batch_group_size = sum(training_util.get_batch_size(batch) for batch in batch_group)
            self._batches_this_epoch += 1
            self._cumulative_batch_group_size += batch_group_size

            if (self._batches_this_epoch - 1) % self._batch_size_interval == 0:
                average = self._cumulative_batch_group_size / self._batches_this_epoch
                logger.info(f"current batch size: {batch_group_size} mean batch size: {average}")
                self.add_train_scalar("current_batch_size", batch_group_size)
                self.add_train_scalar("mean_batch_size", average)

    def reset_epoch(self) -> None:
        self._cumulative_batch_group_size = 0
        self._batches_this_epoch = 0

    def should_log_this_batch(self) -> bool:
        return self.get_batch_num_total() % self._summary_interval == 0

    def should_log_histograms_this_batch(self) -> bool:
        return (
            self._histogram_interval is not None
            and self.get_batch_num_total() % self._histogram_interval == 0
        )

    def add_train_scalar(self, name: str, value: float, timestep: int = None) -> None:
        timestep = timestep or self.get_batch_num_total()
        # get the scalar
        if self._train_log is not None:
            self._train_log.add_scalar(name, self._item(value), timestep)

    def add_train_histogram(self, name: str, values: torch.Tensor) -> None:
        if self._train_log is not None:
            if isinstance(values, torch.Tensor):
                values_to_write = values.cpu().data.numpy().flatten()
                self._train_log.add_histogram(name, values_to_write, self.get_batch_num_total())

    def add_validation_scalar(self, name: str, value: float, timestep: int = None) -> None:
        timestep = timestep or self.get_batch_num_total()
        if self._validation_log is not None:
            self._validation_log.add_scalar(name, self._item(value), timestep)

    def log_parameter_and_gradient_statistics(self, model: Model, batch_grad_norm: float) -> None:
        """
        Send the mean and std of all parameters and gradients to tensorboard, as well
        as logging the average gradient norm.
        """
        if self._should_log_parameter_statistics:
            # Log parameter values to Tensorboard
            for name, param in model.named_parameters():
                if param.data.numel() > 0:
                    self.add_train_scalar("parameter_mean/" + name, param.data.mean())
                if param.data.numel() > 1:
                    self.add_train_scalar("parameter_std/" + name, param.data.std())
                if param.grad is not None:
                    if param.grad.is_sparse:

                        grad_data = param.grad.data._values()
                    else:
                        grad_data = param.grad.data

                    # skip empty gradients
                    if torch.prod(torch.tensor(grad_data.shape)).item() > 0:
                        self.add_train_scalar("gradient_mean/" + name, grad_data.mean())
                        if grad_data.numel() > 1:
                            self.add_train_scalar("gradient_std/" + name, grad_data.std())
                    else:
                        # no gradient for a parameter with sparse gradients
                        logger.info("No gradient for %s, skipping tensorboard logging.", name)
            # norm of gradients
            if batch_grad_norm is not None:
                self.add_train_scalar("gradient_norm", batch_grad_norm)

    def log_learning_rates(self, model: Model, optimizer: torch.optim.Optimizer):
        """
        Send current parameter specific learning rates to tensorboard
        """
        if self._should_log_learning_rate:
            # optimizer stores lr info keyed by parameter tensor
            # we want to log with parameter name
            names = {param: name for name, param in model.named_parameters()}
            for group in optimizer.param_groups:
                if "lr" not in group:
                    continue
                rate = group["lr"]
                for param in group["params"]:
                    # check whether params has requires grad or not
                    effective_rate = rate * float(param.requires_grad)
                    self.add_train_scalar("learning_rate/" + names[param], effective_rate)

    def log_histograms(self, model: Model) -> None:
        """
        Send histograms of parameters to tensorboard.
        """
        if not self._histogram_parameters:
            # Avoiding calling this every batch.  If we ever use two separate models with a single
            # writer, this is wrong, but I doubt that will ever happen.
            self._histogram_parameters = set(
                model.get_parameters_for_histogram_tensorboard_logging()
            )
        for name, param in model.named_parameters():
            if name in self._histogram_parameters:
                self.add_train_histogram("parameter_histogram/" + name, param)

    def log_gradient_updates(self, model: Model, param_updates: Dict[str, torch.Tensor]) -> None:
        for name, param in model.named_parameters():
            update_norm = torch.norm(param_updates[name].view(-1))
            param_norm = torch.norm(param.view(-1)).cpu()
            self.add_train_scalar(
                "gradient_update/" + name,
                update_norm / (param_norm + nn_util.tiny_value_of_dtype(param_norm.dtype)),
            )

    def log_metrics(
        self,
        train_metrics: dict,
        val_metrics: dict = None,
        epoch: int = None,
        log_to_console: bool = False,
    ) -> None:
        """
        Sends all of the train metrics (and validation metrics, if provided) to tensorboard.
        """
        metric_names = set(train_metrics.keys())
        if val_metrics is not None:
            metric_names.update(val_metrics.keys())
        val_metrics = val_metrics or {}

        # For logging to the console
        if log_to_console:
            dual_message_template = "%s |  %8.3f  |  %8.3f"
            no_val_message_template = "%s |  %8.3f  |  %8s"
            no_train_message_template = "%s |  %8s  |  %8.3f"
            header_template = "%s |  %-10s"
            name_length = max(len(x) for x in metric_names)
            logger.info(header_template, "Training".rjust(name_length + 13), "Validation")

        for name in metric_names:
            # Log to tensorboard
            train_metric = train_metrics.get(name)
            if train_metric is not None:
                self.add_train_scalar(name, train_metric, timestep=epoch)
            val_metric = val_metrics.get(name)
            if val_metric is not None:
                self.add_validation_scalar(name, val_metric, timestep=epoch)

            # And maybe log to console
            if log_to_console and val_metric is not None and train_metric is not None:
                logger.info(
                    dual_message_template, name.ljust(name_length), train_metric, val_metric
                )
            elif log_to_console and val_metric is not None:
                logger.info(no_train_message_template, name.ljust(name_length), "N/A", val_metric)
            elif log_to_console and train_metric is not None:
                logger.info(no_val_message_template, name.ljust(name_length), train_metric, "N/A")

    def enable_activation_logging(self, model: Model) -> None:
        if self._histogram_interval is not None:
            # To log activation histograms to the forward pass, we register
            # a hook on forward to capture the output tensors.
            # This uses a closure to determine whether to log the activations,
            # since we don't want them on every call.
            for _, module in model.named_modules():
                if not getattr(module, "should_log_activations", False):
                    # skip it
                    continue

                def hook(module_, inputs, outputs):

                    log_prefix = "activation_histogram/{0}".format(module_.__class__)
                    if self.should_log_histograms_this_batch():
                        self.log_activation_histogram(outputs, log_prefix)

                module.register_forward_hook(hook)

    def log_activation_histogram(self, outputs, log_prefix: str) -> None:
        if isinstance(outputs, torch.Tensor):
            log_name = log_prefix
            self.add_train_histogram(log_name, outputs)
        elif isinstance(outputs, (list, tuple)):
            for i, output in enumerate(outputs):
                log_name = "{0}_{1}".format(log_prefix, i)
                self.add_train_histogram(log_name, output)
        elif isinstance(outputs, dict):
            for k, tensor in outputs.items():
                log_name = "{0}_{1}".format(log_prefix, k)
                self.add_train_histogram(log_name, tensor)
        else:
            # skip it
            pass

    def close(self) -> None:
        """
        Calls the `close` method of the `SummaryWriter` s which makes sure that pending
        scalars are flushed to disk and the tensorboard event files are closed properly.
        """
        if self._train_log is not None:
            self._train_log.close()
        if self._validation_log is not None:
            self._validation_log.close()
