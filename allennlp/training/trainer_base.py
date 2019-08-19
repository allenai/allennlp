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

from allennlp.common import Params, Registrable
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.models.model import Model

logger = logging.getLogger(__name__)


class TrainerBase(Registrable):
    """
    The base class for an AllenNLP trainer. It can do pretty much
    anything you want. Your subclass should implement ``train``
    and also probably ``from_params``.
    """
    default_implementation = "default"

    def __init__(self,
                 serialization_dir: str,
                 cuda_device: Union[int, List] = -1) -> None:
        check_for_gpu(cuda_device)

        self._serialization_dir = serialization_dir

        # Configure GPUs:
        if not isinstance(cuda_device, int) and not isinstance(cuda_device, list):
            raise ConfigurationError("Expected an int or list for cuda_device, got {}".format(cuda_device))

        if isinstance(cuda_device, list):
            self._multiple_gpu = True
            self._cuda_devices = cuda_device
        else:
            self._multiple_gpu = False
            self._cuda_devices = [cuda_device]

    def _move_to_gpu(self, model: Model) -> Model:
        if self._cuda_devices[0] != -1:
            return model.cuda(self._cuda_devices[0])
        else:
            return model

    def train(self) -> Dict[str, Any]:
        """
        Train a model and return the results.
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls,   # type: ignore
                    params: Params,
                    serialization_dir: str,
                    recover: bool = False,
                    cache_directory: str = None,
                    cache_prefix: str = None):
        # pylint: disable=arguments-differ
        typ3 = params.get("trainer", {}).pop("type", "default")

        if typ3 == "default":
            # Special logic to keep old from_params behavior.
            from allennlp.training.trainer import Trainer
            from allennlp.training.trainer_pieces import TrainerPieces

            pieces = TrainerPieces.from_params(params, serialization_dir, recover, cache_directory, cache_prefix)  # pylint: disable=no-member
            return Trainer.from_params(model=pieces.model,
                                       serialization_dir=serialization_dir,
                                       iterator=pieces.iterator,
                                       train_data=pieces.train_dataset,
                                       validation_data=pieces.validation_dataset,
                                       params=pieces.params,
                                       validation_iterator=pieces.validation_iterator)
        else:
            klass = TrainerBase.by_name(typ3)
            # Explicit check to prevent recursion.
            is_overriden = klass.from_params.__func__ != TrainerBase.from_params.__func__ # type: ignore
            assert is_overriden, f"Class {klass.__name__} must override `from_params`."
            return klass.from_params(params, serialization_dir, recover, cache_directory, cache_prefix)
