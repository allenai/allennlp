from typing import (
    Union,
    Tuple,
    OrderedDict,
    Dict,
    NamedTuple,
    List,
    Optional,
    Any,
    TYPE_CHECKING,
)

from overrides import overrides
import torch
from torch.nn.utils import clip_grad_norm_

from allennlp.common import Registrable
from allennlp.common.util import int_to_device

if TYPE_CHECKING:
    # To prevent circular imports
    from allennlp.models import Model


StateDictType = Union[Dict[str, torch.Tensor], OrderedDict[str, torch.Tensor]]


class LoadStateDictReturnType(NamedTuple):
    missing_keys: List[str]
    unexpected_keys: List[str]


class DdpWrappedModel:
    def __init__(
        self,
        model: torch.nn.Module,
        local_rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.model = model

    def load_local_state_dict(
        self, state_dict: StateDictType, strict: bool = True
    ) -> LoadStateDictReturnType:
        return self.model.load_state_dict(state_dict, strict=strict)  # type: ignore[arg-type]

    def load_state_dict(
        self, state_dict: StateDictType, strict: bool = True
    ) -> LoadStateDictReturnType:
        return self.model.load_state_dict(state_dict, strict=strict)  # type: ignore[arg-type]

    def local_state_dict(self, *args, **kwargs) -> Optional[StateDictType]:
        return None

    def state_dict(self, *args, **kwargs) -> StateDictType:
        return self.model.state_dict(*args, **kwargs)

    def clip_grad_norm_(self, max_norm: Union[float, int]) -> torch.Tensor:
        return clip_grad_norm_([p for p in self.model.parameters() if p.grad is not None], max_norm)


class DdpWrapper(Registrable):
    """
    A `DdpWrapper` is a generalization of PyTorch's `DistributedDataParallel` class.

    This is primarly used within the :class:`allennlp.training.trainer.GradientDescentTrainer` to allow
    for different DDP implementations, such as FairScale's [`FullyShardedDataParallel`]
    (https://fairscale.readthedocs.io/en/latest/api/nn/fsdp.html).

    In a typical AllenNLP configuration file, `local_rank`, `world_size`, and `cuda_device`
    should not be specified.

    !!! Warning
        This API is experimental and may change in the future.
    """

    default_implementation = "torch"

    def __init__(
        self, local_rank: int = 0, world_size: int = 1, cuda_device: Union[torch.device, int] = -1
    ) -> None:
        self.local_rank = local_rank
        self.world_size = world_size
        self.primary = local_rank == 0
        self.cuda_device = int_to_device(cuda_device)

    def wrap_model(self, model: "Model") -> Tuple["Model", DdpWrappedModel]:
        """
        Wrap the AllenNLP `Model`, returning the original model (possibly on a different device)
        and the wrapper.
        """
        raise NotImplementedError

    def wrap_module(self, module: torch.nn.Module) -> torch.nn.Module:
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

    def __init__(self, find_unused_parameters: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self._ddp_kwargs = {
            "find_unused_parameters": find_unused_parameters,
        }

    @overrides
    def wrap_model(self, model: "Model") -> Tuple["Model", DdpWrappedModel]:
        if self.cuda_device != torch.device("cpu"):
            model = model.cuda(self.cuda_device)
        wrapped_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=None if self.cuda_device == torch.device("cpu") else [self.cuda_device],
            **self._ddp_kwargs,
        )
        # Add hooks to remove the 'module.' prefix from all keys in the state dict returned
        # from the DistrubtedDataParallel model's `state_dict()` method,
        # and add it back on when loading a state dict.
        wrapped_model._register_state_dict_hook(_remove_torch_ddp_prefix)
        wrapped_model._register_load_state_dict_pre_hook(_add_torch_ddp_prefix)
        return model, DdpWrappedModel(
            wrapped_model,
            local_rank=self.local_rank,
            world_size=self.world_size,
        )


@DdpWrapper.register("no_op")
class NoOpDdpWrapper(DdpWrapper):
    """
    This is a dummy wrapper that doesn't actually do anything. It can be useful
    when you want to use a `DdpWrapper` inside a model when creating submodules
    during distributed training, but don't want to use separate logic when you're
    not in distributed training.
    """

    @overrides
    def wrap_model(self, model: "Model") -> Tuple["Model", DdpWrappedModel]:
        if self.cuda_device != torch.device("cpu"):
            model = model.cuda(self.cuda_device)
        return model, DdpWrappedModel(model)


def _add_torch_ddp_prefix(state_dict: StateDictType, prefix: str, *args: Any) -> None:
    for key in list(state_dict.keys()):
        if key.startswith(prefix + "module."):
            continue
        new_key = prefix + "module." + key
        state_dict[new_key] = state_dict[key]
        del state_dict[key]


def _remove_torch_ddp_prefix(
    module: torch.nn.Module, state_dict: StateDictType, prefix: str, *args: Any
) -> None:
    for key in list(state_dict.keys()):
        if not key.startswith(prefix + "module."):
            continue
        new_key = key.replace(prefix + "module.", "", 1)
        state_dict[new_key] = state_dict[key]
        del state_dict[key]
