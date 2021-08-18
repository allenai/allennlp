import torch

from allennlp.training.learning_rate_schedulers import PolynomialDecay
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler


@LearningRateScheduler.register("linear_with_warmup")
class LinearWithWarmup(PolynomialDecay):
    """
    Implements a learning rate scheduler that increases the learning rate to
    `lr` during the first `warmup_steps` steps, and then decreases it to zero
    over the rest of the training steps.

    In practice, this is a wrapper of [`PolynomialDecay`](
    https://docs.allennlp.org/main/api/training/
    learning_rate_schedulers/polynomial_decay/)
    with `power=1` and `end_learning_rate=0`.


    # Parameters
    optimizer : `torch.optim.Optimizer`
        This argument does not get an entry in a configuration file for the
        object.
    num_epochs: `int`
        The number of epochs in the experiment. this does *NOT* get an entry in
        the config.
    num_steps_per_epoch: `int`
        The number of steps per epoch. this does *NOT* get an entry in the
        config.
    warmup_steps : `int`, required
        The number of steps to linearly increase the learning rate.

    # Example

    Config for using the `LinearWithWarmup` Learning Rate Scheduler with
    `warmup_steps` set `100`.

    ```json
    {
        ...
       "trainer":{
            ...
            "learning_rate_scheduler": {
                "type": "linear_with_warmup",
                "warmup_steps":100
            },
            ...
       }
    }
    ```
    Note that you do NOT pass a `optimizer`, `num_epochs`, nor
    `num_steps_per_epoch` key to the Learning rate scheduler.
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
