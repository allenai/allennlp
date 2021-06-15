from typing import Dict, Any, TYPE_CHECKING

from allennlp.training.callbacks.callback import TrainerCallback

if TYPE_CHECKING:
    from allennlp.training.gradient_descent_trainer import GradientDescentTrainer


@TrainerCallback.register("track_epoch_callback")
class TrackEpochCallback(TrainerCallback):
    """
    A callback that you can pass to the `GradientDescentTrainer` to access the current epoch number
    in your model during training. This callback sets `model.epoch`, which can be read inside of
    `model.forward()`. We set `model.epoch = epoch + 1` which now denotes the number of
    completed epochs at a given training state.
    """

    def on_start(
        self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs
    ) -> None:
        super().on_start(trainer, is_primary)
        trainer.model.epoch = 0  # type: ignore[assignment]

    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        trainer.model.epoch = epoch + 1  # type: ignore[assignment]
