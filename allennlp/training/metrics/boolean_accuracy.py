from typing import Optional

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric


@Metric.register("boolean_accuracy")
class BooleanAccuracy(Metric):
    """
    Just checks batch-equality of two tensors and computes an accuracy metric based on that.  This
    is similar to :class:`CategoricalAccuracy`, if you've already done a ``.max()`` on your
    predictions.  If you have categorical output, though, you should typically just use
    :class:`CategoricalAccuracy`.  The reason you might want to use this instead is if you've done
    some kind of constrained inference and don't have a prediction tensor that matches the API of
    :class:`CategoricalAccuracy`, which assumes a final dimension of size ``num_classes``.
    """
    def __init__(self) -> None:
        self._correct_count = 0.
        self._total_count = 0.

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ...).
        gold_labels : ``torch.Tensor``, required.
            A tensor of the same shape as ``predictions``.
        mask: ``torch.Tensor``, optional (default = None).
            A tensor of the same shape as ``predictions``.
        """
        # Get the data from the Variables.
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        if mask is not None:
            # We can multiply by the mask up front, because we're just checking equality below, and
            # this way everything that's masked will be equal.
            predictions = predictions * mask
            gold_labels = gold_labels * mask

        batch_size = predictions.size(0)
        predictions = predictions.view(batch_size, -1)
        gold_labels = gold_labels.view(batch_size, -1)

        # The .prod() here is functioning as a logical and.
        correct = predictions.eq(gold_labels).prod(dim=1).float()
        count = torch.ones(gold_labels.size(0))
        self._correct_count += correct.sum()
        self._total_count += count.sum()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        accuracy = float(self._correct_count) / float(self._total_count)
        if reset:
            self.reset()
        return accuracy

    @overrides
    def reset(self):
        self._correct_count = 0.0
        self._total_count = 0.0
