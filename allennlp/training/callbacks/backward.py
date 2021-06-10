from typing import TYPE_CHECKING
import torch

from allennlp.training.callbacks.callback import TrainerCallback

if TYPE_CHECKING:
    from allennlp.training.gradient_descent_trainer import GradientDescentTrainer


@TrainerCallback.register("mixed_precision_backward")
class MixedPrecisionBackwardCallback(TrainerCallback):
    """
    Performs backpropagation for mixed precision training.
    """

    def on_backward(
        self,
        trainer: "GradientDescentTrainer",
        loss: torch.FloatTensor,
        backward_called: bool,
        **kwargs
    ) -> bool:
        if not backward_called:
            trainer._scaler.scale(loss).backward()  # type: ignore
            return True
        return False
