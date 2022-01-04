from typing import Optional

from fairscale.nn.checkpoint import checkpoint_wrapper

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
    """

    def __init__(self, offload_to_cpu: Optional[bool] = True) -> None:
        self._offload_to_cpu = offload_to_cpu

    def wrap_module(
        self,
        module: nn.Module,
        **kwargs,
    ) -> nn.Module:
        if "offload_to_cpu" not in kwargs and self._offload_to_cpu is not None:
            kwargs["offload_to_cpu"] = self._offload_to_cpu
        return checkpoint_wrapper(module, **kwargs)
