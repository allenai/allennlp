"""
A :class:`~allennlp.training.trainer.Trainer` is responsible for training a
:class:`~allennlp.models.model.Model`.

Typically you might create a configuration file specifying the model and
training parameters and then use :mod:`~allennlp.commands.train`
rather than instantiating a ``Trainer`` yourself.
"""


import logging
from typing import Dict, Any

from allennlp.common import Params, Registrable
from allennlp.common.util import is_master
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

    def __init__(
        self,
        serialization_dir: str,
        cuda_device: int = -1,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:

        check_for_gpu(cuda_device)
        self._serialization_dir = serialization_dir

        if isinstance(cuda_device, list):
            raise ConfigurationError(
                "In allennlp 1.0, the Trainer can only be assigned a single `cuda_device`. "
                "Instead, we use torch's DistributedDataParallel at the command level, meaning "
                "our Trainer always uses a single GPU per process."
            )

        if not isinstance(cuda_device, int):
            raise ConfigurationError("Expected an int for cuda_device, got {}".format(cuda_device))

        if distributed and world_size <= 1:
            raise ConfigurationError(
                "Distributed training can be performed only with more than 1 GPU device. Check "
                "`cuda_device` key in the experiment configuration."
            )

        self.cuda_device = cuda_device

        self._distributed = distributed
        self._rank = rank
        self._master = is_master()
        self._world_size = world_size

    def _move_to_gpu(self, model: Model) -> Model:
        if self.cuda_device != -1:
            return model.cuda(self.cuda_device)
        else:
            return model

    def train(self) -> Dict[str, Any]:
        """
        Train a model and return the results.
        """
        raise NotImplementedError

    @classmethod
    def from_params(  # type: ignore
        cls,
        params: Params,
        serialization_dir: str,
        recover: bool = False,
        cache_directory: str = None,
        cache_prefix: str = None,
    ):

        typ3 = params.get("trainer", {}).pop("type", "default")

        if typ3 == "default":
            # Special logic to keep old from_params behavior.
            from allennlp.training.trainer import Trainer
            from allennlp.training.trainer_pieces import TrainerPieces

            pieces = TrainerPieces.from_params(
                params, serialization_dir, recover, cache_directory, cache_prefix
            )
            return Trainer.from_params(
                model=pieces.model,
                serialization_dir=serialization_dir,
                iterator=pieces.iterator,
                train_data=pieces.train_dataset,
                validation_data=pieces.validation_dataset,
                params=pieces.params,
                validation_iterator=pieces.validation_iterator,
            )
        else:
            klass = TrainerBase.by_name(typ3)
            # Explicit check to prevent recursion.
            is_overriden = (
                klass.from_params.__func__ != TrainerBase.from_params.__func__  # type: ignore
            )
            assert is_overriden, f"Class {klass.__name__} must override `from_params`."
            return klass.from_params(
                params, serialization_dir, recover, cache_directory, cache_prefix
            )
