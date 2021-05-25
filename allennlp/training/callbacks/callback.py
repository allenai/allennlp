from typing import List, Dict, Any, Optional, TYPE_CHECKING

from allennlp.common import Registrable
from allennlp.data import TensorDict


if TYPE_CHECKING:
    from allennlp.training.trainer import GradientDescentTrainer


class TrainerCallback(Registrable):
    """
    A general callback object that handles multiple events.

    This class has `on_batch`, `on_epoch`, and `on_end` methods, corresponding to
    each callback type. Each one receives the state of the wrapper object as `self`.
    This enables easier state sharing between related callbacks.

    Also, this callback type is instantiated with `serialization_dir` and `on_start` is called
    with the trainer instance as an argument. This might be handy in case of callback logging
    and saving its own files next to the config/checkpoints/logs/etc.
    """

    def __init__(self, serialization_dir: str) -> None:
        self.serialization_dir = serialization_dir
        self.trainer: Optional["GradientDescentTrainer"] = None

    def on_start(
        self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs
    ) -> None:
        """
        This callback hook is called before the training is started.
        """
        self.trainer = trainer

    def on_batch(
        self,
        trainer: "GradientDescentTrainer",
        batch_inputs: List[TensorDict],
        batch_outputs: List[Dict[str, Any]],
        batch_metrics: Dict[str, Any],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_primary: bool = True,
        batch_grad_norm: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        This callback hook is called after the end of each batch.
        """
        pass

    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        """
        This callback hook is called after the end of each epoch.
        """
        pass

    def on_end(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any] = None,
        epoch: int = None,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        """
        This callback hook is called after the final training epoch.
        """
        pass


TrainerCallback.register("null")(TrainerCallback)
