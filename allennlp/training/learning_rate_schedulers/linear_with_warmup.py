from overrides import overrides
import torch

from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler


@LearningRateScheduler.register("linear_with_warmup")
class LinearWithWarmup(LearningRateScheduler):
    """
    Implements a learning rate scheduler that increases the learning rate to `lr` during the first
    `warmup_steps` steps, and then decreases it to zero over the rest of the training steps.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        num_steps_per_epoch: int = None,
        warmup_steps: int = 100,
        last_epoch: int = -1
    ) -> None:
        self.warmup_steps = warmup_steps
        self.num_steps = num_epochs * num_steps_per_epoch
        super().__init__(optimizer, last_epoch=last_epoch)

    @overrides
    def step(self, metric: float = None) -> None:
        pass

    def step_batch(self, batch_num_total: int = None) -> None:
        if batch_num_total is None:
            self.last_epoch += 1  # type: ignore
        else:
            self.last_epoch = batch_num_total
        for param_group, learning_rate in zip(self.optimizer.param_groups, self.get_values()):
            param_group["lr"] = learning_rate

    def get_values(self):
        step = max(self.last_epoch, 0)
        if step < self.warmup_steps:
            scale = (step / self.warmup_steps)
        else:
            fraction_complete = (step - self.warmup_steps) / (self.num_steps - self.warmup_steps)
            scale = 1 - fraction_complete

        return [scale * lr for lr in self.base_values]
