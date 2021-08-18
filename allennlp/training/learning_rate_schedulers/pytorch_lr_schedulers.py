"""
Wrappers for PyTorch's Learning Rate Schedulers so that they work with AllenNLP
"""
from typing import List, Union

import torch
from allennlp.training.optimizers import Optimizer
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import (
    LearningRateScheduler,
    _PyTorchLearningRateSchedulerWrapper,
    _PyTorchLearningRateSchedulerWithMetricsWrapper,
)


@LearningRateScheduler.register("step")
class StepLearningRateScheduler(_PyTorchLearningRateSchedulerWrapper):
    """
    Wrapper for [PyTorch's `StepLR`](
    https://pytorch.org/docs/stable/generated/
    torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR
    ).

    Registered as a `LearningRateScheduler` with name "step".  The "optimizer"
    argument does not get an entry in a configuration file for the object.

    # Parameters
    optimizer : `torch.optim.Optimizer`
        This argument does not get an entry in a configuration file for the
        object.
    step_size : `int`
        Period of learning rate decay.
    gamma: `float`, optional (default = `0.1`)
        Multiplicative factor of learning rate decay.
    last_epoch : `int`, optional (default=`-1`)
        The index of the last epoch. This is used when restarting.


    # Example

    Config for using the `StepLearningRateScheduler` Learning Rate Scheduler
    with `step_size` of `100` and `gamma` set `0.2`.

    ```json
    {
        ...
       "trainer":{
            ...
            "learning_rate_scheduler": {
                "type": "step",
                "step_size": 100,
                "gamma": 0.2
            },
            ...
       }
    }
    ```
    Note that you do NOT pass a `optimizer` key to the Learning rate scheduler.

    """

    def __init__(
        self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1
    ) -> None:
        """

        Args:

        """

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch
        )
        super().__init__(lr_scheduler)


@LearningRateScheduler.register("multi_step")
class MultiStepLearningRateScheduler(_PyTorchLearningRateSchedulerWrapper):
    """
    Wrapper for [PyTorch's `MultiStepLR`](
    https://pytorch.org/docs/stable/
    generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR
    ).

    Registered as a `LearningRateScheduler` with name "multi_step".  The "optimizer" argument does
    not get an entry in a configuration file for the object.

    # Example

    Config for using the `MultiStepLearningRateScheduler` Learning Rate
    Scheduler with `milestones` of `[10,20,40]` and `gamma` set `0.2`.

    ```json
    {
        ...
       "trainer":{
            ...
            "learning_rate_scheduler": {
                "type": "multi_step",
                "milestones": [10,20,40],
                "gamma": 0.2
            },
            ...
       }
    }
    ```
    Note that you do NOT pass a `optimizer` key to the Learning rate scheduler.
    """

    def __init__(
        self, optimizer: Optimizer, milestones: List[int], gamma: float = 0.1, last_epoch: int = -1
    ) -> None:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=milestones, gamma=gamma, last_epoch=last_epoch
        )
        super().__init__(lr_scheduler)


@LearningRateScheduler.register("exponential")
class ExponentialLearningRateScheduler(_PyTorchLearningRateSchedulerWrapper):
    """
    Wrapper for [PyTorch's `ExponentialLR`](
    https://pytorch.org/docs/stable/generated/
    torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR).

    Registered as a `LearningRateScheduler` with name "exponential".  The "optimizer" argument does
    not get an entry in a configuration file for the object.

    # Example

    Config for using the `ExponentialLearningRateScheduler` Learning Rate
    Scheduler with `gamma` set `0.2`.

    ```json
    {
        ...
       "trainer":{
            ...
            "learning_rate_scheduler": {
                "type": "exponential",
                "gamma": 0.2
            },
            ...
       }
    }
    ```
    Note that you do NOT pass a `optimizer` key to the Learning rate scheduler.
    """

    def __init__(self, optimizer: Optimizer, gamma: float = 0.1, last_epoch: int = -1) -> None:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=gamma, last_epoch=last_epoch
        )
        super().__init__(lr_scheduler)


@LearningRateScheduler.register("reduce_on_plateau")
class ReduceOnPlateauLearningRateScheduler(_PyTorchLearningRateSchedulerWithMetricsWrapper):
    """
    Wrapper for [PyTorch's `ReduceLROnPlateau`](https://pytorch.org/docs/stable/generated/
    torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau).

    Registered as a `LearningRateScheduler` with name "reduce_on_plateau".  The
    "optimizer" argument does not get an entry in a configuration file for the
    object.

     # Example

    Config for using the `ReduceOnPlateauLearningRateScheduler` Learning Rate
    Scheduler with the following `init` arguments:

    * `mode="max"`
    * `factor=0.2`
    * `patience=5`
    * `threshold=5e-3`
    * `threshold_mode="abs"`
    * `cooldown=2`
    * `min_lr=1e-12`
    * `eps=1e-10`

    ```json
    {
        ...
       "trainer":{
            ...
            "learning_rate_scheduler": {
                "type": "reduce_on_plateau",
                "mode": "max",
                "factor": 0.2,
                "patience": 5,
                "threshold": 5e-3,
                "threshold_mode": "abs",
                "cooldown": 2,
                "min_lr": 1e-12,
                "eps": 1e-10
            },
            ...
       }
    }
    ```
    Note that you do NOT pass a `optimizer` key to the Learning rate scheduler.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        verbose: bool = False,
        threshold_mode: str = "rel",
        threshold: float = 1e-4,
        cooldown: int = 0,
        min_lr: Union[float, List[float]] = 0,
        eps: float = 1e-8,
    ) -> None:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            verbose=verbose,
            threshold_mode=threshold_mode,
            threshold=threshold,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
        )
        super().__init__(lr_scheduler)
