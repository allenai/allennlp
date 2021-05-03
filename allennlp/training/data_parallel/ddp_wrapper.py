from typing import Union

from overrides import overrides
import torch

from allennlp.common import Registrable
from allennlp.common.util import int_to_device
from allennlp.models import Model


class DdpWrapper(Registrable):
    """
    A `DdpWrapper` is a generalization of PyTorch's `DistributedDataParallel` class.

    This is primarly used within the :class:`allennlp.training.trainer.GradientDescentTrainer` to allow
    for different DDP implementations, such as FairScale's [`FullyShardedDataParallel`]
    (https://fairscale.readthedocs.io/en/latest/api/nn/fsdp.html).
    """

    default_implementation = "torch"

    def __init__(self, model: Model, cuda_device: Union[torch.device, int] = -1) -> None:
        self.model = model
        self.cuda_device = int_to_device(cuda_device)

    def get_wrapped_model(self) -> torch.nn.Module:
        """
        Get the wrapped model.
        """
        raise NotImplementedError


@DdpWrapper.register("torch")
class TorchDdpWrapper(DdpWrapper):
    """
    The default implementation of `DdpWrapper`, which is just a thin wrapper
    around PyTorch's `DistributedDataParallel`.
    """

    def __init__(
        self,
        model: Model,
        cuda_device: Union[torch.device, int] = -1,
        find_unused_parameters: bool = True,
    ) -> None:
        super().__init__(model, cuda_device=cuda_device)
        if (
            self.cuda_device != torch.device("cpu")
            and next(self.model.parameters()).device != self.cuda_device
        ):
            self.model = self.model.cuda(self.cuda_device)
        self._wrapped_model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=None if self.cuda_device == torch.device("cpu") else [self.cuda_device],
            find_unused_parameters=find_unused_parameters,
        )

    @overrides
    def get_wrapped_model(self) -> torch.nn.Module:
        return self._wrapped_model
