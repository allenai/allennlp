import torch

from allennlp.training.learning_rate_schedulers import PolynomialDecay
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler


@LearningRateScheduler.register("linear_with_warmup")
class LinearWithWarmup(PolynomialDecay):
    """
    Implements a learning rate scheduler that increases the learning rate to `lr` during the first
    `warmup_steps` steps, and then decreases it to zero over the rest of the training steps.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        num_steps_per_epoch: int,
        warmup_steps: int = 100,
        last_epoch: int = -1,
    ) -> None:
        super().__init__(
            optimizer,
            num_epochs,
            num_steps_per_epoch,
            power=1.0,
            warmup_steps=warmup_steps,
            end_learning_rate=0.0,
            last_epoch=last_epoch,
        )
