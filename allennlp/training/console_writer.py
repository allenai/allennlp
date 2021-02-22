from typing import Dict, List
from numbers import Number
import logging

import torch
from allennlp.models.model import Model

from allennlp.training.log_writer import LogWriter

logger = logging.getLogger(__name__)


class ConsoleWriter(LogWriter):
    """
    Class that handles console logging.

    # Parameters

    summary_interval : `int`, optional (default = `100`)
        Most statistics will be written out only every this many batches.

    distribution_interval : `int`, optional (default = `None`)
        If provided, distributions of model parameters will be written out
        every this many batches.

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

    def add_train_scalar(self, name: str, value: float, timestep: int = None) -> None:
        assert self.get_batch_num_total is not None
        timestep = timestep or self.get_batch_num_total()
        # header_template = "%s |  %-10s"
        logger.info(f'"{name}" : "{value}" (timestep {timestep})')

    def add_train_tensor(self, name: str, values: torch.Tensor) -> None:
        logger.info(f"\"{name}\" (Shape : {' x '.join([str(x) for x in values.shape])})")
        torch.set_printoptions(threshold=2)
        logger.info(f"{values}")
        torch.set_printoptions(threshold=1000)

    def _log_fields(self, fields: Dict, log_prefix: str = ""):

        for key, val in fields.items():
            if isinstance(val, dict):
                self._log_fields(val, log_prefix + "/" + key)
            elif isinstance(val, torch.Tensor):
                self.add_train_tensor(log_prefix + "/" + key, val)
            elif isinstance(val, Number):
                # This is helpful for a field like `FlagField`.
                self.add_train_scalar(log_prefix + "/" + key, val)  # type: ignore
            elif isinstance(val, List):
                logger.info(f'Field : "{key}" : (Length : {len(val)} of type "{type(val[0])}")')
            elif isinstance(val, str):
                logger.info(f'Field : "{key}"')
                logger.info("{:20.20} ...".format(val))
            else:
                # We do not want to log about the absence of a distribution
                # for this field every single time.
                pass

    def log_metrics(
        self,
        train_metrics: dict,
        val_metrics: dict = None,
        epoch: int = None,
    ) -> None:
        """
        Sends all of the train metrics (and validation metrics, if provided) to tensorboard/console.
        """
        metric_names = set(train_metrics.keys())
        if val_metrics is not None:
            metric_names.update(val_metrics.keys())
        val_metrics = val_metrics or {}

        dual_message_template = "%s |  %8.3f  |  %8.3f"
        no_val_message_template = "%s |  %8.3f  |  %8s"
        no_train_message_template = "%s |  %8s  |  %8.3f"
        header_template = "%s |  %-10s"
        name_length = max(len(x) for x in metric_names)
        logger.info(header_template, "Training".rjust(name_length + 13), "Validation")

        for name in sorted(metric_names):
            train_metric = train_metrics.get(name)
            val_metric = val_metrics.get(name)

            if val_metric is not None and train_metric is not None:
                logger.info(
                    dual_message_template, name.ljust(name_length), train_metric, val_metric
                )
            elif val_metric is not None:
                logger.info(no_train_message_template, name.ljust(name_length), "N/A", val_metric)
            elif train_metric is not None:
                logger.info(no_val_message_template, name.ljust(name_length), train_metric, "N/A")

    def enable_activation_logging(self, model: Model) -> None:
        if self._distribution_interval is not None:
            logger.info("Activation logging is not available for ConsoleWriter.")

    def log_activation_distribution(self, outputs, log_prefix: str) -> None:
        logger.info("Activation logging is not available for ConsoleWriter.")
