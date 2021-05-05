from typing import Union, Tuple

from fairscale.nn import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, auto_wrap, wrap
from overrides import overrides
import torch

from allennlp.nn.parallel.ddp_wrapper import DdpWrapper
from allennlp.models import Model


@DdpWrapper.register("fairscale_fsdp")
class FairScaleFsdpWrapper(DdpWrapper):
    """
    A `DdpWrapper` for FairScale's [`FullyShardedDataParallel`]
    (https://fairscale.readthedocs.io/en/latest/api/nn/fsdp.html).
    """

    def __init__(
        self,
        cuda_device: Union[torch.device, int] = -1,
        mixed_precision: bool = False,
        reshard_after_forward: bool = True,
        cpu_offload: bool = False,
        flatten_parameters: bool = True,
    ) -> None:
        super().__init__(cuda_device=cuda_device)
        self._fsdp_kwargs = {
            "compute_device": self.cuda_device,
            "mixed_precision": mixed_precision,
            "reshard_after_forward": reshard_after_forward,
            "cpu_offload": cpu_offload,
            "flatten_parameters": flatten_parameters,
        }

    @overrides
    def wrap_model(self, model: Model) -> Tuple[Model, torch.nn.Module]:
        wrapped_model = FSDP(
            model,
            **self._fsdp_kwargs,
        )
        if self.cuda_device != torch.device("cpu"):
            wrapped_model = wrapped_model.cuda()
        return model, wrapped_model

    @overrides
    def wrap_module(self, module: torch.nn.Module, recursive: bool = False) -> torch.nn.Module:
        with enable_wrap(wrapper_cls=FSDP, **self._fsdp_kwargs):
            if recursive:
                wrapped_module = auto_wrap(module)
            else:
                wrapped_module = wrap(module)
        return wrapped_module
