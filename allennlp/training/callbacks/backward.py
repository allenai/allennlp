from typing import Dict, TYPE_CHECKING
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
        batch_outputs: Dict[str, torch.Tensor],
        backward_called: bool,
        **kwargs
    ) -> bool:
        if backward_called:
            raise OnBackwardException()
        trainer._scaler.scale(batch_outputs["loss"]).backward()  # type: ignore
        return True


class OnBackwardException(Exception):
    """
    The exception type raised if an `on_backward` callback
    attempts to call `backward` when `backward_called` is `True`.
    """

    def __init__(self, message="") -> None:
        super().__init__(
            "Backpropagation has already been performed"
            "and the computation graph has been erased, so"
            "calling `loss.backward` is not permitted. " + message
        )
