import warnings
from typing import Dict, Optional, Callable
from numbers import Number
import logging
import os

from tensorboardX import SummaryWriter
import torch

from allennlp.models.model import Model

from allennlp.training.log_writer import LogWriter

logger = logging.getLogger(__name__)


class TensorBoardWriter(LogWriter):
    """
    Class that handles TensorBoard logging.

    # Parameters

    serialization_dir : `str`, optional (default = `None`)
        If provided, this is where the TensorBoard logs will be written.

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "tensorboard_writer", it gets passed in separately.

    summary_interval : `int`, optional (default = `100`)
        Most statistics will be written out only every this many batches.

    distribution_interval : `int`, optional (default = `None`)
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

    should_log_inputs : `bool`, optional (default = `False`)
        Whether to log model inputs. Setting it to `True` will log inputs
        on the tensorboard only.

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
        distribution_interval: Optional[int] = None,
        batch_size_interval: Optional[int] = None,
        should_log_parameter_statistics: bool = True,
        should_log_learning_rate: bool = False,
        should_log_inputs: bool = False,
        get_batch_num_total: Callable[[], int] = None,
        **kwargs,
    ) -> None:

        if "histogram_interval" in kwargs:
            warnings.warn(
                "`histogram_interval` is deprecated." "Use `distribution_interval` instead.",
                DeprecationWarning,
            )
            distribution_interval = kwargs.pop("histogram_interval")

        super().__init__(
            serialization_dir,
            summary_interval,
            distribution_interval,
            batch_size_interval,
            should_log_parameter_statistics,
            should_log_learning_rate,
            should_log_inputs,
            get_batch_num_total,
        )

        if self._serialization_dir is not None:
            # Create log directories prior to creating SummaryWriter objects
            # in order to avoid race conditions during distributed training.
            train_ser_dir = os.path.join(self._serialization_dir, "log", "train")
            os.makedirs(train_ser_dir, exist_ok=True)
            self._train_log = SummaryWriter(train_ser_dir)
            val_ser_dir = os.path.join(self._serialization_dir, "log", "validation")
            os.makedirs(val_ser_dir, exist_ok=True)
            self._validation_log = SummaryWriter(val_ser_dir)
        else:
            self._train_log = self._validation_log = None

    def add_train_scalar(self, name: str, value: float, timestep: int = None) -> None:
        assert self.get_batch_num_total is not None
        timestep = timestep or self.get_batch_num_total()
        # get the scalar
        if self._train_log is not None:
            self._train_log.add_scalar(name, self._item(value), timestep)

    def add_train_tensor(self, name: str, values: torch.Tensor) -> None:
        assert self.get_batch_num_total is not None
        if self._train_log is not None:
            if isinstance(values, torch.Tensor):
                values_to_write = values.cpu().data.numpy().flatten()
                self._train_log.add_histogram(name, values_to_write, self.get_batch_num_total())

    def add_train_histogram(self, name: str, values: torch.Tensor) -> None:
        """
        This function is added for backwards compatibility, and simply calls
        `add_train_tensor`.
        """
        self.add_train_tensor(name, values)

    def _log_fields(self, fields: Dict, log_prefix: str = ""):

        for key, val in fields.items():
            if isinstance(val, dict):
                self._log_fields(val, log_prefix + "/" + key)
            elif isinstance(val, torch.Tensor):
                self.add_train_tensor(log_prefix + "/" + key, val)
            elif isinstance(val, Number):
                # This is helpful for a field like `FlagField`.
                self.add_train_scalar(log_prefix + "/" + key, val)  # type: ignore
            else:
                # We do not want to log about the absence of a histogram
                # for this field every single time.
                pass

    def add_validation_scalar(self, name: str, value: float, timestep: int = None) -> None:
        assert self.get_batch_num_total is not None
        timestep = timestep or self.get_batch_num_total()
        if self._validation_log is not None:
            self._validation_log.add_scalar(name, self._item(value), timestep)

    def log_metrics(
        self,
        train_metrics: dict,
        val_metrics: dict = None,
        epoch: int = None,
    ) -> None:
        """
        Sends all of the train metrics (and validation metrics, if provided) to tensorboard.
        """
        metric_names = set(train_metrics.keys())
        if val_metrics is not None:
            metric_names.update(val_metrics.keys())
        val_metrics = val_metrics or {}

        for name in sorted(metric_names):
            # Log to tensorboard
            train_metric = train_metrics.get(name)
            if train_metric is not None:
                self.add_train_scalar(name, train_metric, timestep=epoch)
            val_metric = val_metrics.get(name)
            if val_metric is not None:
                self.add_validation_scalar(name, val_metric, timestep=epoch)

    def enable_activation_logging(self, model: Model) -> None:
        if self._distribution_interval is not None:
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
                    if self.should_log_distributions_this_batch():
                        self.log_activation_distribution(outputs, log_prefix)

                module.register_forward_hook(hook)

    def log_activation_distribution(self, outputs, log_prefix: str) -> None:
        if isinstance(outputs, torch.Tensor):
            log_name = log_prefix
            self.add_train_tensor(log_name, outputs)
        elif isinstance(outputs, (list, tuple)):
            for i, output in enumerate(outputs):
                log_name = "{0}_{1}".format(log_prefix, i)
                self.add_train_tensor(log_name, output)
        elif isinstance(outputs, dict):
            for k, tensor in outputs.items():
                log_name = "{0}_{1}".format(log_prefix, k)
                self.add_train_tensor(log_name, tensor)
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
