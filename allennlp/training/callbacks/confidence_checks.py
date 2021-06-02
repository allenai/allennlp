from typing import List, Dict, Any, Optional, TYPE_CHECKING

from allennlp.training.callbacks.callback import TrainerCallback
from allennlp.data import TensorDict
from allennlp.confidence_checks.normalization_bias_verification import NormalizationBiasVerification


if TYPE_CHECKING:
    from allennlp.training.gradient_descent_trainer import GradientDescentTrainer


# `sanity_checks` is deprecated and will be removed.
@TrainerCallback.register("sanity_checks")
@TrainerCallback.register("confidence_checks")
class ConfidenceChecksCallback(TrainerCallback):
    """
    Performs model confidence checks.

    Checks performed:

    * `NormalizationBiasVerification` for detecting invalid combinations of
       bias and normalization layers.
       See `allennlp.confidence_checks.normalization_bias_verification` for more details.

    Note: Any new confidence checks should also be added to this callback.
    """

    def on_start(
        self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs
    ) -> None:
        self.trainer = trainer
        if is_primary:
            self._verification = NormalizationBiasVerification(self.trainer._pytorch_model)
            # Register the hooks that perform the verification before training starts.
            self._verification.register_hooks()

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
        if not is_primary:
            return None

        # We destroy the hooks after the first batch, since we only want to
        # perform this check once.
        if epoch == 0 and batch_number == 1 and is_training:
            self._verification.destroy_hooks()
            detected_pairs = self._verification.collect_detections()
            if len(detected_pairs) > 0:
                raise ConfidenceCheckError(
                    "The NormalizationBiasVerification check failed. See logs for more details."
                )


class ConfidenceCheckError(Exception):
    """
    The error type raised when a confidence check fails.
    """

    def __init__(self, message) -> None:
        super().__init__(
            message
            + "\nYou can disable these checks by setting the trainer parameter `run_confidence_checks` to `False`."
        )
