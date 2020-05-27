from overrides import overrides
import torch

from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler


@LearningRateScheduler.register("noam")
class NoamLR(LearningRateScheduler):
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first `warmup_steps` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.

    Registered as a `LearningRateScheduler` with name "noam".

    # Parameters

    optimizer : `torch.optim.Optimizer`
        This argument does not get an entry in a configuration file for the object.
    model_size : `int`, required.
        The hidden size parameter which dominates the number of parameters in your model.
    warmup_steps : `int`, required.
        The number of steps to linearly increase the learning rate.
    factor : `float`, optional (default = `1.0`).
        The overall scale factor for the learning rate decay.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model_size: int,
        warmup_steps: int,
        factor: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.model_size = model_size
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
        step = max(self.last_epoch, 1)
        scale = self.factor * (
            self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )

        return [scale for _ in range(len(self.base_values))]
