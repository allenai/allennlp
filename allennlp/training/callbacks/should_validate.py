from typing import Dict, Any, TYPE_CHECKING, Optional

from allennlp.training.callbacks.callback import TrainerCallback

if TYPE_CHECKING:
    from allennlp.training.gradient_descent_trainer import GradientDescentTrainer


@TrainerCallback.register("should_validate_callback")
class ShouldValidateCallback(TrainerCallback):
    """
    A callback that you can pass to the `GradientDescentTrainer` to change the frequency of
    validation during training. If `validation_start` is not `None`, validation will not occur until
    `validation_start` epochs have elapsed. If `validation_interval` is not `None`, validation will
    run every `validation_interval` number of epochs epochs.
    """

    def __init__(
        self,
        serialization_dir: str,
        validation_start: Optional[int] = None,
        validation_interval: Optional[int] = None,
    ) -> None:
        super().__init__(serialization_dir)
        self._validation_start = validation_start
        self._validation_interval = validation_interval

    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        if self._validation_start is not None and epoch < self._validation_start:
            trainer._should_validate_this_epoch = False
        elif self._validation_interval is not None and epoch % self._validation_interval != 0:
            trainer._should_validate_this_epoch = False
        else:
            trainer._should_validate_this_epoch = True
