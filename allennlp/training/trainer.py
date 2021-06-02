import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch.optim.lr_scheduler

from allennlp.common import Registrable
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common.util import int_to_device

logger = logging.getLogger(__name__)


@dataclass
class TrainerCheckpoint:
    model_state: Dict[str, Any]
    trainer_state: Dict[str, Any]


class Trainer(Registrable):
    """
    The base class for an AllenNLP trainer. It can do pretty much
    anything you want. Your subclass should implement `train`
    and also probably `from_params`.
    """

    default_implementation = "gradient_descent"

    def __init__(
        self,
        serialization_dir: str = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        distributed: bool = False,
        local_rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if cuda_device is None:
            from torch import cuda

            if cuda.device_count() > 0:
                cuda_device = 0
            else:
                cuda_device = -1

        check_for_gpu(cuda_device)

        if serialization_dir is None:
            import tempfile

            self._serialization_dir = tempfile.mkdtemp()
        else:
            self._serialization_dir = serialization_dir
        # Ensure serialization directory exists.
        os.makedirs(self._serialization_dir, exist_ok=True)

        if isinstance(cuda_device, list):
            raise ConfigurationError(
                "In AllenNLP 1.0, the Trainer can only be assigned a single `cuda_device`. "
                "Instead, we use torch's DistributedDataParallel at the command level, meaning "
                "our Trainer always uses a single GPU per process."
            )

        if distributed and world_size <= 1:
            raise ConfigurationError(
                "Distributed training can be performed only with more than 1 device. Check "
                "`cuda_device` key in the experiment configuration."
            )

        self.cuda_device = int_to_device(cuda_device)

        self._distributed = distributed
        self._rank = local_rank
        self._primary = self._rank == 0
        self._world_size = world_size

    def train(self) -> Dict[str, Any]:
        """
        Train a model and return the results.
        """
        raise NotImplementedError

    def get_checkpoint_state(self) -> TrainerCheckpoint:
        """
        Returns a tuple of (model state, training state), where training state could have several
        internal components (e.g., for an, optimizer, learning rate scheduler, etc.).
        """
        raise NotImplementedError

    def get_best_weights_path(self) -> Optional[str]:
        """Returns the path to file containing the current best weights."""
        return None
