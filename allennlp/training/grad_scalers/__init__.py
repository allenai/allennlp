from allennlp.training.grad_scalers.grad_scaler import GradScaler

try:
    from allennlp.training.grad_scalers.fairscale_sharded_grad_scaler import (
        FairScaleShardedGradScaler,
    )
except ModuleNotFoundError as exc:
    if exc.name == "fairscale":
        pass
    else:
        raise
