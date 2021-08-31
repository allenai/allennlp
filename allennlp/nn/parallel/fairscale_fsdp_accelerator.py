import os
import logging
from typing import Tuple, Union, Optional, TYPE_CHECKING, List, Any, Dict, Sequence

from fairscale.nn import FullyShardedDataParallel as FS_FSDP
from fairscale.nn.wrap import enable_wrap, wrap
from fairscale.nn.misc import FlattenParamsWrapper

# from fairscale.optim.grad_scaler import GradScaler
from fairscale.experimental.optim.dynamic_loss_scaler import DynamicLossScaler
from overrides import overrides
import torch
from torch.cuda import amp

from allennlp.nn.parallel.sharded_module_mixin import ShardedModuleMixin
from allennlp.nn.parallel.ddp_accelerator import (
    DdpAccelerator,
    DdpWrappedModel,
    StateDictType,
    LoadStateDictReturnType,
)

if TYPE_CHECKING:
    # To prevent circular imports
    from allennlp.models import Model

logger = logging.getLogger(__name__)


class DeepSpeedLossScaler:
    """
    Class that manages dynamic loss scaling.  It is recommended to use :class:`DynamicLossScaler`
    indirectly, by supplying ``dynamic_loss_scale=True`` to the constructor of
    :class:`FP16_Optimizer`.  However, it's important to understand how :class:`DynamicLossScaler`
    operates, because the default options can be changed using the
    the ``dynamic_loss_args`` argument to :class:`FP16_Optimizer`'s constructor.
    Loss scaling is designed to combat the problem of underflowing gradients encountered at long
    times when training fp16 networks.  Dynamic loss scaling begins by attempting a very high loss
    scale.  Ironically, this may result in OVERflowing gradients.  If overflowing gradients are
    encountered, :class:`DynamicLossScaler` informs :class:`FP16_Optimizer` that an overflow has
    occurred.
    :class:`FP16_Optimizer` then skips the update step for this particular iteration/minibatch,
    and :class:`DynamicLossScaler` adjusts the loss scale to a lower value.
    If a certain number of iterations occur without overflowing gradients detected,
    :class:`DynamicLossScaler` increases the loss scale once more.
    In this way :class:`DynamicLossScaler` attempts to "ride the edge" of
    always using the highest loss scale possible without incurring overflow.
    Args:
        init_scale (float, optional, default=2**32):  Initial loss scale attempted by :class:`DynamicLossScaler.`
        scale_factor (float, optional, default=2.0):  Factor used when adjusting the loss scale.
        If an overflow is encountered, the loss scale is readjusted to loss scale/``scale_factor``.
        If ``scale_window`` consecutive iterations take place without an overflow,
        the loss scale is readjusted to loss_scale*``scale_factor``.
        scale_window (int, optional, default=1000):  Number of consecutive iterations without an overflow to wait
        before increasing the loss scale.
    """

    def __init__(
        self,
        init_scale=2 ** 32,
        scale_factor=2.0,
        scale_window=1000,
        min_scale=1,
        delayed_shift=1,
        consecutive_hysteresis=False,
    ):
        self.cur_scale = init_scale
        self.cur_iter = 0
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale
        self.delayed_shift = delayed_shift
        self.cur_hysteresis = delayed_shift
        self.consecutive_hysteresis = consecutive_hysteresis

        self.cur_overflow = False

    def _has_overflow(self, optimizer: torch.optim.Optimizer):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None and DynamicLossScaler._has_inf_or_nan(param.grad.data):
                    return True
        return False

    @classmethod
    def _has_inf_or_nan(x: torch.Tensor):
        try:
            # if x is half, the .float() incurs an additional deep copy, but it's necessary if
            # Pytorch's .sum() creates a one-element tensor of the same type as x
            # (which is true for some recent version of pytorch).
            cpu_sum = float(x.float().sum())
            # More efficient version that can be used if .sum() returns a Python scalar
            # cpu_sum = float(x.sum())
        except RuntimeError as instance:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum in [float("inf"), -float("inf")] or cpu_sum != cpu_sum:
                return True
            return False

    def update(self):
        if self.cur_overflow:
            if self.delayed_shift == 1 or self.cur_hysteresis == 1:
                self.cur_scale = max(self.cur_scale / self.scale_factor, self.min_scale)
            else:
                self.cur_hysteresis -= 1
            self.last_overflow_iter = self.cur_iter
        else:
            if self.consecutive_hysteresis:
                self.cur_hysteresis = self.delayed_shift
            if (self.cur_iter - self.last_overflow_iter) % self.scale_window == 0:
                if not self.consecutive_hysteresis:
                    self.cur_hysteresis = self.delayed_shift
                self.cur_scale *= self.scale_factor
        self.cur_iter += 1
        # reset
        self.cur_overflow = False

    @property
    def loss_scale(self):
        return self.cur_scale

    def unscale_(self, optimizer: torch.optim.Optimizer):
        with torch.no_grad():
            for group in optimizer.param_groups:
                for param in group["params"]:
                    param.grad.data.mul_(1.0 / self.loss_scale)

    def step(self, optimizer: torch.optim.Optimizer):
        has_overflow = self._has_overflow(optimizer)
        self.cur_overflow = has_overflow
        if has_overflow:
            # We skip the iteration.
            logger.warning("FP16 dynamic loss scale overflow!")
        else:
            self.unscale_(optimizer)
            optimizer.step()

    def scale(self, loss: torch.Tensor):
        return loss * self.loss_scale


class _FSDP(FS_FSDP, ShardedModuleMixin):
    """
    Same as FairScale's [`FullyShardedDataParallel`]
    (https://fairscale.readthedocs.io/en/latest/api/nn/fsdp.html) but also implements
    the mixin methods from :class:`allennlp.nn.parallel.sharded_module_mixin.ShardedModuleMixin`.
    """

    @overrides
    def get_original_module(self) -> torch.nn.Module:
        module = self.module
        if isinstance(module, FlattenParamsWrapper):
            module = module.module
        return module


class FairScaleFsdpWrappedModel(DdpWrappedModel):
    """
    The wrapped model type returned from [`FairScaleFsdpWrappedModel.wrap_model`](#wrap_model).
    """

    @staticmethod
    @overrides
    def consolidate_sharded_state(
        sharded_state_files: Sequence[Union[str, os.PathLike]]
    ) -> StateDictType:
        shard_weights: List[StateDictType] = []
        shard_metadata: List[Dict[str, Any]] = []
        for path in sharded_state_files:
            shard_state = torch.load(path, map_location="cpu")
            shard_weights.append(shard_state["weights"])
            shard_metadata.append(shard_state["metadata"])
        return _FSDP.consolidate_shard_weights(shard_weights, shard_metadata)

    @overrides
    def load_state_dict(
        self, state_dict: StateDictType, strict: bool = True
    ) -> LoadStateDictReturnType:
        return self.model.load_local_state_dict(state_dict["weights"], strict=strict)  # type: ignore[operator]

    @overrides
    def state_dict(self, *args, **kwargs) -> StateDictType:
        weights = self.model.local_state_dict(*args, **kwargs)  # type: ignore[operator]
        metadata = self.model.local_metadata_dict()
        return {"weights": weights, "metadata": metadata}

    @overrides
    def clip_grad_norm_(self, max_norm: Union[float, int]) -> torch.Tensor:
        return self.model.clip_grad_norm_(max_norm)  # type: ignore[operator]

    @overrides
    def init_grad_scaler(self) -> amp.GradScaler:
        return DynamicLossScaler()


@DdpAccelerator.register("fairscale_fsdp")
class FairScaleFsdpAccelerator(DdpAccelerator):
    """
    A :class:`allennlp.nn.parallel.ddp_accelerator.DdpAccelerator` for FairScale's [`FullyShardedDataParallel`]
    (https://fairscale.readthedocs.io/en/latest/api/nn/fsdp.html).

    To save memory while initializing a model, you should call [`.wrap_module()`](#wrap_module) on submodules
    as they're created.

    See the :class:`allennlp.modules.transformer.t5.T5` class for an example of how to use this.
    """

    def __init__(
        self,
        *,
        mixed_precision: bool = False,
        reshard_after_forward: bool = True,
        flatten_parameters: bool = True,
        local_rank: Optional[int] = None,
        world_size: Optional[int] = None,
        cuda_device: Union[torch.device, int] = -1,
    ) -> None:
        super().__init__(local_rank=local_rank, world_size=world_size, cuda_device=cuda_device)
        self._fsdp_kwargs = {
            "compute_device": self.cuda_device,
            "mixed_precision": mixed_precision,
            "reshard_after_forward": reshard_after_forward,
            "flatten_parameters": flatten_parameters,
        }
        if mixed_precision:
            self._fsdp_kwargs["move_params_to_cpu"] = True
            self._fsdp_kwargs["clear_autocast_cache"] = True

    @overrides
    def wrap_model(self, model: "Model") -> Tuple["Model", DdpWrappedModel]:
        wrapped_model = _FSDP(
            model,
            **self._fsdp_kwargs,
        )
        if not self._fsdp_kwargs["mixed_precision"] and self.cuda_device != torch.device("cpu"):
            wrapped_model = wrapped_model.cuda()
        # `_FSDP._lazy_init()` may have been called already on submodules that were wrapped
        # (through `wrap_module()`), leading those submodules to think they are root submodules.
        # So we need to call `_FSDP._reset_lazy_init()` on any of these now.
        for module in wrapped_model.modules():
            if isinstance(module, _FSDP):
                module._reset_lazy_init()
        return model, FairScaleFsdpWrappedModel(
            wrapped_model,
            local_rank=self.local_rank,
            world_size=self.world_size,
        )

    @overrides
    def wrap_module(self, module: torch.nn.Module) -> torch.nn.Module:
        with enable_wrap(wrapper_cls=_FSDP, **self._fsdp_kwargs):
            wrapped_module = wrap(module)
        return wrapped_module
