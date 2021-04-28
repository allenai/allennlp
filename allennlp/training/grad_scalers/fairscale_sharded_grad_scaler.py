from fairscale.optim.grad_scaler import ShardedGradScaler as _FairScaleShardedGradScaler
from allennlp.training.grad_scalers.grad_scaler import GradScaler


@GradScaler.register("fairscale")
class FairScaleShardedGradScaler(_FairScaleShardedGradScaler, GradScaler):
    """
    FairScale's `ShardedGradScaler`.
    """
