from allennlp.nn.parallel.sharded_module_mixin import ShardedModuleMixin
from allennlp.nn.parallel.ddp_wrapper import (
    DdpWrapper,
    DdpWrappedModel,
    TorchDdpWrapper,
)

try:
    from allennlp.nn.parallel.fairscale_fsdp_wrapper import (
        FairScaleFsdpWrapper,
        FairScaleFsdpWrappedModel,
    )
except ModuleNotFoundError as exc:
    # FairScale is an optional dependency.
    if exc.name == "fairscale":
        pass
    else:
        raise
