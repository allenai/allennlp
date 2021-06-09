from typing import TYPE_CHECKING
import torch

from allennlp.training.callbacks.callback import TrainerCallback

if TYPE_CHECKING:
    from allennlp.training.gradient_descent_trainer import GradientDescentTrainer


class BackwardCallback(TrainerCallback):
    def on_backward(
        self, trainer: "GradientDescentTrainer", loss: torch.FloatTensor, **kwargs
    ) -> None:
        """
        This callback hook performs backpropagation and allows for gradient manipulation.
        """
        raise NotImplementedError


@TrainerCallback.register("vanilla_backward")
class VanillaBackwardCallback(BackwardCallback):
    """
    Performs vanilla backpropagation.
    """

    def on_backward(
        self, trainer: "GradientDescentTrainer", loss: torch.FloatTensor, **kwargs
    ) -> None:
        loss.backward()


@TrainerCallback.register("mixed_precision_backward")
class MixedPrecisionBackwardCallback(BackwardCallback):
    """
    Performs backpropagation for mixed precision training.
    """

    def on_backward(
        self, trainer: "GradientDescentTrainer", loss: torch.FloatTensor, **kwargs
    ) -> None:
        trainer._scaler.scale(loss).backward()  # type: ignore


class BackwardCallbackError(Exception):
    """
    The error type raised when multiple callbacks passed to a trainer
    implement `on_backward`.
    """

    def __init__(self, message) -> None:
        super().__init__(message)
