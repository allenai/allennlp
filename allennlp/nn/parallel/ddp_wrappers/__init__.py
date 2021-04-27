from allennlp.nn.parallel.ddp_wrappers.ddp_wrapper import DdpWrapper, TorchDdpWrapper

try:
    from allennlp.nn.parallel.ddp_wrappers.fairscale_sharded_ddp_wrapper import (
        FairScaleShardedDdpWrapper,
    )
except ModuleNotFoundError as exc:
    if exc.name == "fairscale":
        pass
    else:
        raise
