from typing import Optional

from fairscale.nn.checkpoint import checkpoint_wrapper
from overrides import overrides
import torch.nn as nn

from allennlp.nn.checkpoint.checkpoint_wrapper import CheckpointWrapper


@CheckpointWrapper.register("fairscale")
class FairScaleCheckpointWrapper(CheckpointWrapper):
    """
    Provides [FairScale]
    (https://fairscale.readthedocs.io/en/latest/api/nn/checkpoint/checkpoint_activations.html)'s
    activation/gradient checkpointing functionality.

    The parameters and their defaults are the same as they are in FairScale, and
    any of them can be overriden on a per-module basis by passing the corresponding parameter
    to `.wrap_module()`.

    This can also be used in conjunction with the
    :class:`allennlp.nn.parallel.fairscale_fsdp_accelerator.FairScaleFsdpAccelerator`.
    See the [T5 implementation](/api/modules/transformer/t5/) for an example
    of how to use the two together.

    !!! Note
        If using the `FairScaleFsdpAccelerator`, you need to set `maintain_forward_counter` to `True`.
        For convenience, if `maintain_forward_counter` is not set, internally it will be
        set to `True` if training in a distributed setup, or `False` otherwise.
    """

    def __init__(
        self, offload_to_cpu: Optional[bool] = True, maintain_forward_counter: Optional[bool] = None
    ) -> None:
        self._offload_to_cpu = offload_to_cpu
        if maintain_forward_counter is None:
            from allennlp.common.util import is_distributed

            # Better to assume we need this in the distributed case, since we definitely
            # need this when the model is wrapped with FairScale's FSDP.
            self._maintain_forward_counter = is_distributed()
        else:
            self._maintain_forward_counter = maintain_forward_counter

    @overrides
    def wrap_module(
        self,
        module: nn.Module,
        **kwargs,
    ) -> nn.Module:
        if "offload_to_cpu" not in kwargs and self._offload_to_cpu is not None:
            kwargs["offload_to_cpu"] = self._offload_to_cpu
        if "maintain_forward_counter" not in kwargs and self._maintain_forward_counter is not None:
            kwargs["maintain_forward_counter"] = self._maintain_forward_counter
        return checkpoint_wrapper(module, **kwargs)
