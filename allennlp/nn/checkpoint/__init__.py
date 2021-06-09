from allennlp.nn.checkpoint.checkpoint_wrapper import CheckpointWrapper, TorchCheckpointWrapper

try:
    from allennlp.nn.checkpoint.fairscale_checkpoint_wrapper import FairScaleCheckpointWrapper
except ModuleNotFoundError as exc:
    # FairScale is an optional dependency.
    if exc.name == "fairscale":
        pass
    else:
        raise
