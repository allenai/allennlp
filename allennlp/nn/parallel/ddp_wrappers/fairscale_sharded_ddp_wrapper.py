from typing import List, Optional

from overrides import overrides
from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import ShardedDataParallel
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.nn.parallel.ddp_wrappers.ddp_wrapper import DdpWrapper


@DdpWrapper.register("fairscale")
class FairScaleShardedDdpWrapper(DdpWrapper):
    """
    Wraps FairScale's `ShardedDataParallel`.

    !!! Note
        When training with AMP enabled, setting `reduce_fp16` to `True` will
        probably improve performance.

    """

    def __init__(self, auto_refresh_trainable: bool = True, reduce_fp16: bool = False) -> None:
        self._auto_refresh_trainable = auto_refresh_trainable
        self._reduce_fp16 = reduce_fp16

    @overrides
    def initialize(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        device_ids: Optional[List[torch.device]] = None,
    ) -> torch.nn.Module:
        if not isinstance(optimizer, OSS):
            raise ConfigurationError(
                "optimizer is required to be an instance of FairScale's OSS class"
            )
        return ShardedDataParallel(
            model,
            optimizer,
            auto_refresh_trainable=self._auto_refresh_trainable,
            reduce_fp16=self._reduce_fp16,
        )
