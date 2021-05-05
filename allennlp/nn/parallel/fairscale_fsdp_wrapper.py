from typing import Tuple, Union, Optional

from fairscale.nn import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, auto_wrap, wrap
from overrides import overrides
import torch

from allennlp.nn.parallel.ddp_wrapper import (
    DdpWrapper,
    DdpWrappedModel,
    StateDictType,
    LoadStateDictReturnType,
)
from allennlp.models import Model


class FairScaleFsdpWrappedModel(DdpWrappedModel):
    def __init__(self, model: torch.nn.Module, **kwargs) -> None:
        super().__init__(model, **kwargs)

    @overrides
    def load_local_state_dict(
        self, state_dict: StateDictType, strict: bool = True
    ) -> LoadStateDictReturnType:
        return self.model.load_local_state_dict(state_dict, strict=strict)

    @overrides
    def local_state_dict(self, *args, **kwargs) -> Optional[StateDictType]:
        return self.model.local_state_dict(*args, **kwargs)

    @overrides
    def clip_grad_norm_(self, max_norm: Union[float, int]) -> torch.Tensor:
        return self.model.clip_grad_norm_(max_norm)


@DdpWrapper.register("fairscale_fsdp")
class FairScaleFsdpWrapper(DdpWrapper):
    """
    A `DdpWrapper` for FairScale's [`FullyShardedDataParallel`]
    (https://fairscale.readthedocs.io/en/latest/api/nn/fsdp.html).
    """

    def __init__(
        self,
        mixed_precision: bool = False,
        reshard_after_forward: bool = True,
        cpu_offload: bool = False,
        flatten_parameters: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._fsdp_kwargs = {
            "compute_device": self.cuda_device,
            "mixed_precision": mixed_precision,
            "reshard_after_forward": reshard_after_forward,
            "cpu_offload": cpu_offload,
            "flatten_parameters": flatten_parameters,
        }

    @overrides
    def wrap_model(self, model: Model) -> Tuple[Model, DdpWrappedModel]:
        wrapped_model = FSDP(
            model,
            **self._fsdp_kwargs,
        )
        if self.cuda_device != torch.device("cpu"):
            wrapped_model = wrapped_model.cuda()
        return model, FairScaleFsdpWrappedModel(
            wrapped_model, local_rank=self.local_rank, world_size=self.world_size
        )

    @overrides
    def wrap_module(self, module: torch.nn.Module, recursive: bool = False) -> torch.nn.Module:
        with enable_wrap(wrapper_cls=FSDP, **self._fsdp_kwargs):
            if recursive:
                wrapped_module = auto_wrap(module)
            else:
                wrapped_module = wrap(module)
        return wrapped_module
