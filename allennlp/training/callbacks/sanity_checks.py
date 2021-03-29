from typing import List, Dict, Any, Optional, TYPE_CHECKING

from allennlp.training.callbacks.callback import TrainerCallback
from allennlp.data import TensorDict
from allennlp.sanity_checks.normalization_bias_verification import NormalizationBiasVerification


if TYPE_CHECKING:
    from allennlp.training.trainer import GradientDescentTrainer


@TrainerCallback.register("sanity_checks")
class SanityChecksCallback(TrainerCallback):
    """
    Performs model sanity checks.

    Checks performed:

    * `NormalizationBiasVerification` for detecting invalid combinations of
       bias and normalization layers.
       See `allennlp.sanity_checks.normalization_bias_verification` for more details.

    Note: Any new sanity checks should also be added to this callback.
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
            assert (
                len(detected_pairs) == 0
            ), "The NormalizationBiasVerification check failed. See logs for more details."
