import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING

import torch

from allennlp.training.callbacks.callback import TrainerCallback
from allennlp.training.util import get_train_and_validation_metrics
from allennlp.data import TensorDict

if TYPE_CHECKING:
    from allennlp.training.trainer import GradientDescentTrainer


logger = logging.getLogger(__name__)


@TrainerCallback.register("console_logger")
class ConsoleLoggerCallback(TrainerCallback):
    def __init__(
        self,
        serialization_dir: str,
        should_log_inputs: bool = False,
    ) -> None:
        super().__init__(serialization_dir)
        self._should_log_inputs = should_log_inputs

    def on_batch(
        self,
        trainer: "GradientDescentTrainer",
        batch_inputs: List[TensorDict],
        batch_outputs: List[Dict[str, Any]],
        batch_metrics: Dict[str, Any],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_primary: bool = True,
        batch_grad_norm: Optional[float] = None,
        **kwargs,
    ) -> None:

        if not is_primary:
            return None

        # We only want to do this for the first batch in the first epoch.
        if batch_number == 1 and epoch == 0 and self._should_log_inputs:
            logger.info("Batch inputs")
            for b, batch in enumerate(batch_inputs):
                self._log_fields(batch, log_prefix="batch_input")  # type: ignore

    def _log_fields(self, fields: Dict, log_prefix: str = ""):
        for key, val in fields.items():
            key = log_prefix + "/" + key
            if isinstance(val, dict):
                self._log_fields(val, key)
            elif isinstance(val, torch.Tensor):
                torch.set_printoptions(threshold=2)
                logger.info("%s (Shape: %s)\n%s", key, " x ".join([str(x) for x in val.shape]), val)
                torch.set_printoptions(threshold=1000)
            elif isinstance(val, List):
                logger.info('Field : "%s" : (Length %d of type "%s")', key, len(val), type(val[0]))
            elif isinstance(val, str):
                logger.info('Field : "{}" : "{:20.20} ..."'.format(key, val))
            else:
                logger.info('Field : "%s" : %s', key, val)

    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        if not is_primary:
            return None

        train_metrics, val_metrics = get_train_and_validation_metrics(metrics)

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
