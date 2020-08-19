from overrides import overrides
import torch

from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler


@LearningRateScheduler.register("polynomial_decay")
class PolynomialDecay(LearningRateScheduler):
    """
    Implements polynomial decay Learning rate scheduling. The learning rate is first
    linearly increased for the first `warmup_steps` training steps. Then it is decayed for
    `total_steps` - `warmup_steps` from the initial learning rate to `end_learning_rate` using a polynomial
    of degree `power`.

    Formally,

    `lr` = (`initial_lr` - `end_learning_rate`) *
           ((`total_steps` - `steps`)/(`total_steps` - `warmup_steps`)) ** `power`

    # Parameters

    total_steps: `int`, required
        The total number of steps to adjust the learning rate for.
    warmup_steps : `int`, required
        The number of steps to linearly increase the learning rate.
    power : `float`, optional (default = `1.0`)
        The power of the polynomial used for decaying.
    end_learning_rate : `float`, optional (default = `0.0`)
        Final learning rate to decay towards.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        num_steps_per_epoch: int,
        power=1.0,
        warmup_steps=0,
        end_learning_rate=0.0,
        last_epoch: int = -1,
    ):
        super().__init__(optimizer, last_epoch)

        self.power = power
        self.warmup_steps = warmup_steps
        self.total_steps = num_epochs * num_steps_per_epoch
        self.end_learning_rate = end_learning_rate

        self.steps = 0

        self.step_batch(0)

    @overrides
    def get_values(self):
        if self.warmup_steps > 0 and self.steps < self.warmup_steps:
            f = self.steps / self.warmup_steps
            return [f * lr for lr in self.base_values]

        if self.steps >= self.total_steps:
            return [self.end_learning_rate for _ in self.base_values]

        current_decay_steps = self.total_steps - self.steps
        total_decay_steps = self.total_steps - self.warmup_steps
        f = (current_decay_steps / total_decay_steps) ** self.power
        return [
            f * (lr - self.end_learning_rate) + self.end_learning_rate for lr in self.base_values
        ]

    @overrides
    def step(self, metric: float = None) -> None:
        pass

    @overrides
    def step_batch(self, batch_num_total: int = None) -> None:
        if batch_num_total is None:
            self.steps += 1
        else:
            self.steps = batch_num_total

        for param_group, lr in zip(self.optimizer.param_groups, self.get_values()):
            param_group[self.param_group_field] = lr
