from collections import deque, defaultdict
from typing import Any, Dict, List, Optional, Set, Union, Deque
import logging

import torch

from allennlp.common import Registrable
from allennlp.data import TensorDict
from allennlp.nn import util as nn_util
from allennlp.training.optimizers import Optimizer
from allennlp.training import util as training_util
from allennlp.models.model import Model


logger = logging.getLogger(__name__)


class LogWriter(Registrable):
    """
    Class that handles trainer logging.

    # Parameters

    serialization_dir : `str`
        The training serialization directory.

        In a typical AllenNLP configuration file, this parameter does not get an entry in the
        file, it gets passed in separately.

    model : `Model`
        The model being trained.

        In a typical AllenNLP configuration file, this parameter does not get an entry in the
        file, it gets passed in separately.

    optimizer : `Optimizer`
        The optimizer used to train the model.

        In a typical AllenNLP configuration file, this parameter does not get an entry in the
        file, it gets passed in separately.

    summary_interval : `int`, optional (default = `100`)
        Most statistics will be written out only every this many batches.

    distribution_interval : `int`, optional (default = `None`)
        When this parameter is specified, the following additional logging is enabled
        every this many batches:

            * Distributions of model parameters
            * The ratio of parameter update norm to parameter norm
            * Distribution of layer activations

        The layer activations are logged for any modules in the `Model` that have
        the attribute `should_log_activations` set to `True`.

        Logging distributions requires a number of GPU-CPU copies during training and is typically
        slow, so we recommend logging distributions relatively infrequently.

        !!! Note
            Only Modules that return tensors, tuples of tensors or dicts
            with tensors as values currently support activation logging.

    batch_size_interval : `int`, optional, (default = `None`)
        If defined, how often to log the average batch size.

    should_log_parameter_statistics : `bool`, optional (default = `True`)
        Whether to log parameter statistics (mean and standard deviation of parameters and
        gradients). If `True`, parameter stats are logged every `summary_interval` batches.

    should_log_learning_rate : `bool`, optional (default = `False`)
        Whether to log (parameter-specific) learning rate.
        If `True`, learning rates are logged every `summary_interval` batches.

    batch_loss_moving_average_count : `int`, optional (default = `100`)
        The length of the moving average for batch loss.

    """

    def __init__(
        self,
        serialization_dir: str,
        model: Model,
        optimizer: Optimizer,
        summary_interval: int = 100,
        distribution_interval: Optional[int] = None,
        batch_size_interval: Optional[int] = None,
        should_log_parameter_statistics: bool = True,
        should_log_learning_rate: bool = False,
        batch_loss_moving_average_count: int = 100,
    ):
        self._serialization_dir = serialization_dir
        self._model = model
        self._optimizer = optimizer
        self._summary_interval = summary_interval
        self._distribution_interval = distribution_interval
        self._batch_size_interval = batch_size_interval
        self._should_log_parameter_statistics = should_log_parameter_statistics
        self._should_log_learning_rate = should_log_learning_rate
        self._cumulative_batch_group_size = 0
        self._distribution_parameters: Optional[Set[str]] = None
        self._module_hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._batch_num_total: int = 0
        self._batch_loss_moving_average_count = batch_loss_moving_average_count
        self._batch_loss_moving_sum: Dict[str, float] = defaultdict(float)
        self._batch_loss_moving_items: Dict[str, Deque[float]] = defaultdict(deque)

        self._enable_activation_logging()

    def log_scalars(
        self,
        scalars: Dict[str, Union[int, float]],
        log_prefix: str = "",
        epoch: Optional[int] = None,
    ) -> None:
        """
        Required to be implemented by subclasses.

        Defines how batch or epoch scalar metrics are logged.
        """
        raise NotImplementedError

    def log_tensors(
        self, tensors: Dict[str, torch.Tensor], log_prefix: str = "", epoch: Optional[int] = None
    ) -> None:
        """
        Required to be implemented by subclasses.

        Defines how batch or epoch tensor metrics are logged.
        """
        raise NotImplementedError

    def log_inputs(self, inputs: List[TensorDict], log_prefix: str = "") -> None:
        """
        Can be optionally implemented by subclasses.

        Defines how batch inputs are logged. This is called once at the start of each epoch.
        """
        pass

    def log_batch(
        self,
        batch_grad_norm: Optional[float],
        metrics: Dict[str, float],
        batch_group: List[TensorDict],
        param_updates: Optional[Dict[str, torch.Tensor]],
        batch_number: int,
    ) -> None:
        """
        Called every batch to perform all of the logging that is due.
        """
        if batch_number <= 1:  # batch_number is usually 1-indexed
            self._cumulative_batch_group_size = 0
            self.log_inputs(batch_group)

        if self._should_log_this_batch():
            if self._should_log_parameter_statistics:
                self._log_parameter_and_gradient_statistics(batch_grad_norm)

            if self._should_log_learning_rate:
                self._log_learning_rates()

            # Now collect per-batch metrics to log.
            metrics_to_log: Dict[str, float] = {}
            for key in ("batch_loss", "batch_reg_loss"):
                if key not in metrics:
                    continue
                value = metrics[key]
                metrics_to_log[key] = value
                # Update and add moving average.
                self._batch_loss_moving_sum[key] += value
                self._batch_loss_moving_items[key].append(value)
                if len(self._batch_loss_moving_items[key]) > self._batch_loss_moving_average_count:
                    self._batch_loss_moving_sum[key] -= self._batch_loss_moving_items[key].popleft()
                metrics_to_log[f"{key}_mov_avg"] = self._batch_loss_moving_sum[key] / len(
                    self._batch_loss_moving_items[key]
                )

            self.log_scalars(
                metrics_to_log,
                log_prefix="train",
            )

        if self.should_log_distributions_this_batch():
            assert param_updates is not None
            self._log_distributions()
            self._log_gradient_updates(param_updates)

        if self._batch_size_interval:
            # We're assuming here that `log_batch` will get called every batch, and only every
            # batch.  This is true with our current usage of this code (version 1.0); if that
            # assumption becomes wrong, this code will break.
            batch_group_size = sum(training_util.get_batch_size(batch) for batch in batch_group)  # type: ignore
            self._cumulative_batch_group_size += batch_group_size
            if batch_number % self._batch_size_interval == 0:
                average = self._cumulative_batch_group_size / batch_number
                self.log_scalars(
                    {"batch_size": batch_group_size, "mean_batch_size": average}, log_prefix="train"
                )

        self._batch_num_total += 1

    def log_epoch(
        self,
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any],
        epoch: int,
    ) -> None:
        """
        Called at the end of every epoch to log training and validation metrics.
        """
        self.log_scalars(
            {
                k: v
                for k, v in train_metrics.items()
                if isinstance(v, (int, float))
                if "_memory_MB" not in k  # W&B gives us much better system metrics
            },
            log_prefix="train",
            epoch=epoch,
        )
        self.log_scalars(
            {k: v for k, v in val_metrics.items() if isinstance(v, (int, float))},
            log_prefix="validation",
            epoch=epoch,
        )

    def should_log_distributions_next_batch(self) -> bool:
        return (
            self._distribution_interval is not None
            and (self._batch_num_total + 2) % self._distribution_interval == 0
        )

    def should_log_distributions_this_batch(self) -> bool:
        return (
            self._distribution_interval is not None
            and (self._batch_num_total + 1) % self._distribution_interval == 0
        )

    def _enable_activation_logging(self) -> None:
        if self._distribution_interval is not None:
            # To log activation histograms to the forward pass, we register
            # a hook on forward to capture the output tensors.
            # This uses a closure to determine whether to log the activations,
            # since we don't want them on every call.
            for _, module in self._model.named_modules():
                if not getattr(module, "should_log_activations", False):
                    # skip it
                    continue

                def hook(module_, inputs, outputs):
                    if self.should_log_distributions_this_batch():
                        self._log_activation_distribution(outputs, str(module_.__class__))

                self._module_hook_handles.append(module.register_forward_hook(hook))

    def _should_log_this_batch(self) -> bool:
        return self._batch_num_total % self._summary_interval == 0

    def _log_activation_distribution(self, outputs: Any, module_name: str) -> None:
        activations_to_log: Dict[str, torch.Tensor] = {}
        if isinstance(outputs, torch.Tensor):
            log_name = module_name
            activations_to_log[log_name] = outputs
        elif isinstance(outputs, (list, tuple)):
            for i, output in enumerate(outputs):
                if isinstance(output, torch.Tensor):
                    log_name = "{0}_{1}".format(module_name, i)
                    activations_to_log[log_name] = output
        elif isinstance(outputs, dict):
            for k, output in outputs.items():
                log_name = "{0}_{1}".format(module_name, k)
                if isinstance(output, torch.Tensor):
                    activations_to_log[log_name] = output

        if activations_to_log:
            self.log_tensors(activations_to_log, log_prefix="activation_histogram")

    def _log_parameter_and_gradient_statistics(self, batch_grad_norm: float = None) -> None:
        parameter_mean_scalars: Dict[str, float] = {}
        parameter_std_scalars: Dict[str, float] = {}
        gradient_mean_scalars: Dict[str, float] = {}
        gradient_std_scalars: Dict[str, float] = {}
        # Log parameter values to TensorBoard
        for name, param in self._model.named_parameters():
            if param.data.numel() > 0:
                parameter_mean_scalars[name] = param.data.mean().item()
            if param.data.numel() > 1:
                parameter_std_scalars[name] = param.data.std().item()
            if param.grad is not None:
                if param.grad.is_sparse:
                    grad_data = param.grad.data._values()
                else:
                    grad_data = param.grad.data

                # skip empty gradients
                if torch.prod(torch.tensor(grad_data.shape)).item() > 0:
                    gradient_mean_scalars[name] = grad_data.mean().item()
                    if grad_data.numel() > 1:
                        gradient_std_scalars[name] = grad_data.std().item()
                else:
                    # no gradient for a parameter with sparse gradients
                    logger.info("No gradient for %s, skipping logging.", name)
        self.log_scalars(parameter_mean_scalars, log_prefix="parameter_mean")
        self.log_scalars(parameter_std_scalars, log_prefix="parameter_std")
        self.log_scalars(gradient_mean_scalars, log_prefix="gradient_mean")
        self.log_scalars(gradient_std_scalars, log_prefix="gradient_std")
        # norm of gradients
        if batch_grad_norm is not None:
            self.log_scalars({"gradient_norm": batch_grad_norm})

    def _log_learning_rates(self):
        # optimizer stores lr info keyed by parameter tensor
        # we want to log with parameter name
        lr_scalars: Dict[str, float] = {}
        names = {param: name for name, param in self._model.named_parameters()}
        for group in self._optimizer.param_groups:
            if "lr" not in group:
                continue
            rate = group["lr"]
            for param in group["params"]:
                # check whether params has requires grad or not
                effective_rate = rate * float(param.requires_grad)
                lr_scalars[names[param]] = effective_rate
        self.log_scalars(lr_scalars, log_prefix="learning_rate")

    def _log_distributions(self) -> None:
        """
        Log distributions of parameters.
        """
        if not self._distribution_parameters:
            # Avoiding calling this every batch.  If we ever use two separate models with a single
            # writer, this is wrong, but I doubt that will ever happen.
            self._distribution_parameters = set(self._model.get_parameters_for_histogram_logging())
        parameters_to_log: Dict[str, torch.Tensor] = {}
        for name, param in self._model.named_parameters():
            if name in self._distribution_parameters:
                parameters_to_log[name] = param
        self.log_tensors(parameters_to_log, log_prefix="parameter_histogram")

    def _log_gradient_updates(self, param_updates: Dict[str, torch.Tensor]) -> None:
        gradient_update_scalars: Dict[str, float] = {}
        for name, param in self._model.named_parameters():
            update_norm = torch.norm(param_updates[name].view(-1))
            param_norm = torch.norm(param.view(-1)).cpu()
            gradient_update_scalars[name] = (
                update_norm / (param_norm + nn_util.tiny_value_of_dtype(param_norm.dtype))
            ).item()
        self.log_scalars(gradient_update_scalars, log_prefix="gradient_update")

    def close(self) -> None:
        """
        Called at the end of training to remove any module hooks and close out any
        other logging resources.
        """
        for handle in self._module_hook_handles:
            handle.remove()
