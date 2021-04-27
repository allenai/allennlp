from typing import List, Optional

from overrides import overrides
import torch

from allennlp.common import Registrable


class DdpWrapper(Registrable):
    """
    A `DdpWrapper` is a generalization of PyTorch's `DistributedDataParallel` class.

    This is primarly used within the :class:`allennlp.training.trainer.GradientDescentTrainer` to allow
    for different DDP implementations, such as FairScale's `ShardedDataParallel`.
    """

    default_implementation = "torch"

    def initialize(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        device_ids: Optional[List[torch.device]] = None,
    ) -> torch.nn.Module:
        """
        Initializes the DDP implementation, returning the wrapped model.
        """
        raise NotImplementedError


@DdpWrapper.register("torch")
class TorchDdpWrapper(DdpWrapper):
    """
    The default implementation of `DdpWrapper`, which is just a thin wrapper
    around PyTorch's `DistributedDataParallel`.
    """

    @overrides
    def initialize(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        device_ids: Optional[List[torch.device]] = None,
    ) -> torch.nn.Module:
        return torch.nn.parallel.DistributedDataParallel(
            model, device_ids=device_ids, find_unused_parameters=True
        )
