from typing import Any, Set, Optional, Callable
import logging
import os

from tensorboardX import SummaryWriter
import torch

from allennlp.models.model import Model

logger = logging.getLogger(__name__)


class TensorboardWriter:
    """
    Wraps a pair of ``SummaryWriter`` instances but is a no-op if they're ``None``.
    Allows Tensorboard logging without always checking for Nones first.
    """
    def __init__(self, train_log: SummaryWriter = None, validation_log: SummaryWriter = None) -> None:
        self._train_log = train_log
        self._validation_log = validation_log

    @staticmethod
    def create(serialization_dir: Optional[str]) -> 'TensorboardWriter':
        if serialization_dir is not None:
            train_log = SummaryWriter(os.path.join(serialization_dir, "log", "train"))
            validation_log = SummaryWriter(os.path.join(serialization_dir, "log", "validation"))
            return TensorboardWriter(train_log, validation_log)
        else:
            return TensorboardWriter()

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


    def log_parameter_and_gradient_statistics(self, # pylint: disable=invalid-name
                                              model: Model,
                                              epoch: int,
                                              batch_grad_norm: float) -> None:
        """
        Send the mean and std of all parameters and gradients to tensorboard, as well
        as logging the average gradient norm.
        """
        # Log parameter values to Tensorboard
        for name, param in model.named_parameters():
            self.add_train_scalar("parameter_mean/" + name,
                                  param.data.mean(),
                                  epoch)
            self.add_train_scalar("parameter_std/" + name, param.data.std(), epoch)
            if param.grad is not None:
                if param.grad.is_sparse:
                    # pylint: disable=protected-access
                    grad_data = param.grad.data._values()
                else:
                    grad_data = param.grad.data

                # skip empty gradients
                if torch.prod(torch.tensor(grad_data.shape)).item() > 0: # pylint: disable=not-callable
                    self.add_train_scalar("gradient_mean/" + name,
                                          grad_data.mean(),
                                          epoch)
                    self.add_train_scalar("gradient_std/" + name,
                                          grad_data.std(),
                                          epoch)
                else:
                    # no gradient for a parameter with sparse gradients
                    logger.info("No gradient for %s, skipping tensorboard logging.", name)
        # norm of gradients
        if batch_grad_norm is not None:
            self.add_train_scalar("gradient_norm",
                                  batch_grad_norm,
                                  epoch)

    def log_learning_rates(self,
                           model: Model,
                           optimizer: torch.optim.Optimizer,
                           batch_num_total: int):
        """
        Send current parameter specific learning rates to tensorboard
        """
        # optimizer stores lr info keyed by parameter tensor
        # we want to log with parameter name
        names = {param: name for name, param in model.named_parameters()}
        for group in optimizer.param_groups:
            if 'lr' not in group:
                continue
            rate = group['lr']
            for param in group['params']:
                # check whether params has requires grad or not
                effective_rate = rate * float(param.requires_grad)
                self.add_train_scalar(
                        "learning_rate/" + names[param],
                        effective_rate,
                        batch_num_total
                )


    def log_histograms(self, model: Model, epoch: int, histogram_parameters: Set[str]) -> None:
        """
        Send histograms of parameters to tensorboard.
        """
        for name, param in model.named_parameters():
            if name in histogram_parameters:
                self.add_train_histogram("parameter_histogram/" + name,
                                         param,
                                         epoch)


    def log_metrics(self,
                    epoch: int,
                    train_metrics: dict,
                    val_metrics: dict = None,
                    log_to_console: bool = True) -> None:
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
            name_length = max([len(x) for x in metric_names])
            logger.info(header_template, "Training".rjust(name_length + 13), "Validation")

        for name in metric_names:
            # Log to tensorboard
            train_metric = train_metrics.get(name)
            if train_metric is not None:
                self.add_train_scalar(name, train_metric, epoch)
            val_metric = val_metrics.get(name)
            if val_metric is not None:
                self.add_validation_scalar(name, val_metric, epoch)

            # And maybe log to console
            if log_to_console and val_metric is not None and train_metric is not None:
                logger.info(dual_message_template, name.ljust(name_length), train_metric, val_metric)
            elif log_to_console and val_metric is not None:
                logger.info(no_train_message_template, name.ljust(name_length), "N/A", val_metric)
            elif log_to_console and train_metric is not None:
                logger.info(no_val_message_template, name.ljust(name_length), train_metric, "N/A")


    def enable_activation_logging(self, model: Model, get_batch_num_total: Callable[[], Optional[int]]) -> None:
        # To log activation histograms to the forward pass, we register
        # a hook on forward to capture the output tensors.
        # This uses a closure to determine whether to log the activations,
        # since we don't want them on every call.
        for _, module in model.named_modules():
            if not getattr(module, 'should_log_activations', False):
                # skip it
                continue

            def hook(module_, inputs, outputs):
                # pylint: disable=unused-argument,cell-var-from-loop
                log_prefix = 'activation_histogram/{0}'.format(module_.__class__)
                batch_num_total = get_batch_num_total()
                if batch_num_total is not None:
                    self.log_activation_histogram(outputs, log_prefix, batch_num_total)
            module.register_forward_hook(hook)

    def log_activation_histogram(self, outputs, log_prefix: str, batch_num_total: int) -> None:
        if isinstance(outputs, torch.Tensor):
            log_name = log_prefix
            self.add_train_histogram(log_name, outputs, batch_num_total)
        elif isinstance(outputs, (list, tuple)):
            for i, output in enumerate(outputs):
                log_name = "{0}_{1}".format(log_prefix, i)
                self.add_train_histogram(log_name, output, batch_num_total)
        elif isinstance(outputs, dict):
            for k, tensor in outputs.items():
                log_name = "{0}_{1}".format(log_prefix, k)
                self.add_train_histogram(log_name, tensor, batch_num_total)
        else:
            # skip it
            pass
