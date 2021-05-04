from typing import Union, Tuple

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

    In a typical AllenNLP configuration file, `cuda_device` should not be specified.
    """

    default_implementation = "torch"

    def __init__(self, cuda_device: Union[torch.device, int] = -1) -> None:
        self.cuda_device = int_to_device(cuda_device)

    def wrap_model(self, model: Model) -> Tuple[Model, torch.nn.Module]:
        """
        Wrap the AllenNLP `Model`, returning the original model (possibly on a different device)
        and the wrapper.
        """
        raise NotImplementedError

    def wrap_module(self, module: torch.nn.Module, recursive: bool = False) -> torch.nn.Module:
        """
        Wrap an individual module. By default this just returns the module,
        but some subclass implementations such as `FairScaleFsdpWrapper` do more.
        """
        return module


@DdpWrapper.register("torch")
class TorchDdpWrapper(DdpWrapper):
    """
    The default implementation of `DdpWrapper`, which is just a thin wrapper
    around PyTorch's `DistributedDataParallel`.
    """

    def __init__(
        self,
        cuda_device: Union[torch.device, int] = -1,
        find_unused_parameters: bool = True,
    ) -> None:
        super().__init__(cuda_device=cuda_device)
        self._ddp_kwargs = {
            "find_unused_parameters": find_unused_parameters,
        }

    @overrides
    def wrap_model(self, model: Model) -> Tuple[Model, torch.nn.Module]:
        if self.cuda_device != torch.device("cpu"):
            model = model.cuda(self.cuda_device)
        wrapped_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=None if self.cuda_device == torch.device("cpu") else [self.cuda_device],
            **self._ddp_kwargs,
        )
        return model, wrapped_model
