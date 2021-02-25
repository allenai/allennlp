from typing import Any, Callable, Dict, List, Optional, Set
import logging

import torch

from allennlp.common.from_params import FromParams
from allennlp.data import TensorDict
from allennlp.nn import util as nn_util
from allennlp.training.optimizers import Optimizer
from allennlp.training import util as training_util
from allennlp.models.model import Model

logger = logging.getLogger(__name__)


class LogWriter(FromParams):
    """
    Class that handles trainer logging.

    # Parameters

    serialization_dir : `str`, optional (default = `None`)
        If provided, this is where the logs will be written.

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "tensorboard_writer", it gets passed in separately.

    summary_interval : `int`, optional (default = `100`)
        Most statistics will be written out only every this many batches.

    distribution_interval : `int`, optional (default = `None`)
        If provided, activation distributions will be written out every this many batches.
        If None, activation distributions will not be written out.
        When this parameter is specified, the following additional logging is enabled:

            * Distributions of model parameters
            * The ratio of parameter update norm to parameter norm
            * Distribution of layer activations

        The layer activations are logged for any modules in the `Model` that have
        the attribute `should_log_activations` set to `True`.  Logging
        distributions requires a number of GPU-CPU copies during training and is typically
        slow, so we recommend logging distributions relatively infrequently.
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
        Whether to log model inputs.

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
    ):
        self._serialization_dir = serialization_dir
        self._summary_interval = summary_interval
        self._distribution_interval = distribution_interval
        self._batch_size_interval = batch_size_interval
        self._should_log_parameter_statistics = should_log_parameter_statistics
        self._should_log_learning_rate = should_log_learning_rate
        self._should_log_inputs = should_log_inputs
        self.get_batch_num_total = get_batch_num_total

        self._cumulative_batch_group_size = 0
        self._batches_this_epoch = 0
        self._distribution_parameters: Optional[Set[str]] = None

    @staticmethod
    def _item(value: Any):
        if hasattr(value, "item"):
            val = value.item()
        else:
            val = value
        return val

    def reset_epoch(self) -> None:
        self._cumulative_batch_group_size = 0
        self._batches_this_epoch = 0

    def should_log_this_batch(self) -> bool:
        assert self.get_batch_num_total is not None
        return self.get_batch_num_total() % self._summary_interval == 0

    def should_log_distributions_next_batch(self) -> bool:
        assert self.get_batch_num_total is not None
        return (
            self._distribution_interval is not None
            and (self.get_batch_num_total() + 1) % self._distribution_interval == 0
        )

    def should_log_distributions_this_batch(self) -> bool:
        assert self.get_batch_num_total is not None
        return (
            self._distribution_interval is not None
            and self.get_batch_num_total() % self._distribution_interval == 0
        )

    def add_train_scalar(self, name: str, value: float, timestep: int = None):
        """
        This function is for how scalar values should be logged.
        """
        return NotImplementedError

    def add_validation_scalar(self, name: str, value: float, timestep: int = None):
        return NotImplementedError

    def add_train_tensor(self, name: str, values: torch.Tensor):
        """
        This function is for how tensor values should be logged.
        """
        return NotImplementedError

    def log_metrics(
        self,
        train_metrics: dict,
        val_metrics: dict = None,
        epoch: int = None,
    ):
        """
        Sends all of the train metrics (and validation metrics, if provided) to tensorboard/console.
        """
        return NotImplementedError

    def enable_activation_logging(self, model: Model):
        return NotImplementedError

    def log_activation_distribution(self, outputs, log_prefix: str):
        return NotImplementedError

    def _log_fields(self, fields: Dict, log_prefix: str = ""):
        return NotImplementedError

    def log_inputs(self, batch_group: List[List[TensorDict]]):
        for b, batch in enumerate(batch_group):
            self._log_fields(batch, log_prefix="batch_input")  # type: ignore

    def log_memory_usage(self, cpu_memory_usage: Dict[int, int], gpu_memory_usage: Dict[int, int]):
        cpu_memory_usage_total = 0.0
        for worker, mem_bytes in cpu_memory_usage.items():
            memory = mem_bytes / (1024 * 1024)
            self.add_train_scalar(f"memory_usage/worker_{worker}_cpu", memory)
            cpu_memory_usage_total += memory
        self.add_train_scalar("memory_usage/cpu", cpu_memory_usage_total)
        for gpu, mem_bytes in gpu_memory_usage.items():
            memory = mem_bytes / (1024 * 1024)
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

        if self.should_log_distributions_this_batch():
            assert param_updates is not None
            self.log_distributions(model)
            self.log_gradient_updates(model, param_updates)

            if self._should_log_inputs:
                self.log_inputs(batch_group)

        if self._batch_size_interval:
            # We're assuming here that `log_batch` will get called every batch, and only every
            # batch.  This is true with our current usage of this code (version 1.0); if that
            # assumption becomes wrong, this code will break.
            batch_group_size = sum(training_util.get_batch_size(batch) for batch in batch_group)  # type: ignore
            self._batches_this_epoch += 1
            self._cumulative_batch_group_size += batch_group_size

            if (self._batches_this_epoch - 1) % self._batch_size_interval == 0:
                average = self._cumulative_batch_group_size / self._batches_this_epoch
                logger.info(f"current batch size: {batch_group_size} mean batch size: {average}")
                self.add_train_scalar("current_batch_size", batch_group_size)
                self.add_train_scalar("mean_batch_size", average)

    def log_distributions(self, model: Model) -> None:
        """
        Send distributions of parameters to tensorboard.
        """
        if not self._distribution_parameters:
            # Avoiding calling this every batch.  If we ever use two separate models with a single
            # writer, this is wrong, but I doubt that will ever happen.
            self._distribution_parameters = set(
                model.get_parameters_for_histogram_tensorboard_logging()
            )
        for name, param in model.named_parameters():
            if name in self._distribution_parameters:
                self.add_train_tensor("parameter_histogram/" + name, param)

    def log_parameter_and_gradient_statistics(
        self, model: Model, batch_grad_norm: float = None
    ) -> None:
        """
        Send the mean and std of all parameters and gradients to tensorboard, as well
        as logging the average gradient norm.
        """
        if self._should_log_parameter_statistics:
            # Log parameter values to TensorBoard
            for name, param in model.named_parameters():
                if param.data.numel() > 0:
                    self.add_train_scalar("parameter_mean/" + name, param.data.mean().item())
                if param.data.numel() > 1:
                    self.add_train_scalar("parameter_std/" + name, param.data.std().item())
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
                        logger.info("No gradient for %s, skipping logging.", name)
            # norm of gradients
            if batch_grad_norm is not None:
                self.add_train_scalar("gradient_norm", batch_grad_norm)

    def log_learning_rates(self, model: Model, optimizer: Optimizer):
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

    def log_gradient_updates(self, model: Model, param_updates: Dict[str, torch.Tensor]) -> None:
        for name, param in model.named_parameters():
            update_norm = torch.norm(param_updates[name].view(-1))
            param_norm = torch.norm(param.view(-1)).cpu()
            self.add_train_scalar(
                "gradient_update/" + name,
                update_norm / (param_norm + nn_util.tiny_value_of_dtype(param_norm.dtype)),
            )

    def close(self) -> None:
        pass
