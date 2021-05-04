from allennlp.nn.parallel.ddp_wrapper import DdpWrapper, TorchDdpWrapper

try:
    from allennlp.nn.parallel.fairscale_fsdp_wrapper import FairScaleFsdpWrapper
except ModuleNotFoundError as exc:
    # FairScale is an optional dependency.
    if exc.name == "fairscale":
        pass
    else:
        raise
