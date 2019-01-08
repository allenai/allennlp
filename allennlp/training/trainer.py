"""
A :class:`~allennlp.training.trainer.Trainer` is responsible for training a
:class:`~allennlp.models.model.Model`.

Typically you might create a configuration file specifying the model and
training parameters and then use :mod:`~allennlp.commands.train`
rather than instantiating a ``Trainer`` yourself.
"""
# pylint: disable=too-many-lines

import logging
from typing import Dict, List, Union, Any

import torch
import torch.optim.lr_scheduler

from allennlp.common import Params, Registrable
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.models.model import Model
from allennlp.nn import util as nn_util
from allennlp.training.tensorboard_writer import TensorboardWriter
from allennlp.training import util as training_util

logger = logging.getLogger(__name__)


class Trainer(Registrable):
    """
    The base class for an AllenNLP trainer.
    The only method you *have* to implement is ``train``.
    """
    default_implementation = "single_task"

    def __init__(self,
                 model: Model,
                 serialization_dir: str,
                 cuda_device: Union[int, List] = -1) -> None:
        check_for_gpu(cuda_device)

        self.model = model
        self._serialization_dir = serialization_dir

        # Set up tensorboard
        self._tensorboard = TensorboardWriter.create(serialization_dir)

        # Configure GPUs:
        if not isinstance(cuda_device, int) and not isinstance(cuda_device, list):
            raise ConfigurationError("Expected an int or list for cuda_device, got {}".format(cuda_device))

        if isinstance(cuda_device, list):
            logger.warning(f"Multiple GPU support is experimental not recommended for use. "
                           "In some cases it may lead to incorrect results or undefined behavior.")
            self._multiple_gpu = True
            self._cuda_devices = cuda_device
        else:
            self._multiple_gpu = False
            self._cuda_devices = [cuda_device]

        if self._cuda_devices[0] != -1:
            self.model = self.model.cuda(self._cuda_devices[0])

        self._warned_tqdm_ignores_underscores = False

    def train(self) -> Dict[str, Any]:
        """
        Train a model and return the results.
        """
        raise NotImplementedError

    def batch_loss(self, batch: torch.Tensor, for_training: bool) -> torch.Tensor:
        """
        Does a forward pass on the given batch and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.
        """
        if self._multiple_gpu:
            output_dict = training_util.data_parallel(batch, self.model, self._cuda_devices)
        else:
            batch = nn_util.move_to_device(batch, self._cuda_devices[0])
            output_dict = self.model(**batch)

        try:
            loss = output_dict["loss"]
            if for_training:
                loss += self.model.get_regularization_penalty()
        except KeyError:
            if for_training:
                raise RuntimeError("The model you are trying to optimize does not contain a"
                                   " 'loss' key in the output of model.forward(inputs).")
            loss = None

        return loss

    @classmethod
    def from_params(cls,   # type: ignore
                    params: Params,
                    serialization_dir: str,
                    recover: bool = False):
        # pylint: disable=arguments-differ
        typ3 = params.pop("type", cls.default_implementation)
        return Trainer.by_name(typ3).from_params(params, serialization_dir, recover)

    def _description_from_metrics(self, metrics: Dict[str, float]) -> str:
        if (not self._warned_tqdm_ignores_underscores and
                    any(metric_name.startswith("_") for metric_name in metrics)):
            logger.warning("Metrics with names beginning with \"_\" will "
                           "not be logged to the tqdm progress bar.")
            self._warned_tqdm_ignores_underscores = True
        return ', '.join(["%s: %.4f" % (name, value) for name, value in
                          metrics.items() if not name.startswith("_")]) + " ||"
