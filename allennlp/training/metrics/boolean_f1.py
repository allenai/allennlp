from typing import Optional

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric
from allennlp.nn.util import ones_like


@Metric.register("boolean_f1")
class BooleanF1(Metric):
    """
    Just checks batch-equality of two tensors and computes an accuracy metric based on that.  This
    is similar to :class:`CategoricalAccuracy`, if you've already done a ``.max()`` on your
    predictions.  If you have categorical output, though, you should typically just use
    :class:`CategoricalAccuracy`.  The reason you might want to use this instead is if you've done
    some kind of constrained inference and don't have a prediction tensor that matches the API of
    :class:`CategoricalAccuracy`, which assumes a final dimension of size ``num_classes``.
    """
    def __init__(self) -> None:
        self._true_positives = 0.
        self._false_positives = 0.
        self._false_negatives = 0.
        self._true_negatives = 0.
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
            predictions = predictions * mask
            gold_labels = gold_labels * mask

        positive_label_mask = gold_labels.eq(1)
        negative_label_mask = gold_labels.eq(0)

        # True Positives: correct positively labeled predictions.
        tp = (predictions == 1) * positive_label_mask
        fp = (predictions == 1) * negative_label_mask
        fn = (predictions == 0) * positive_label_mask
        tn = (predictions == 0) * negative_label_mask

        self._true_positives += tp.sum()
        self._false_positives += fp.sum()
        self._false_negatives += fn.sum()
        self._true_negatives += tn.sum()

        self._total_count = self._true_positives + self._false_positives + self._false_negatives + self._true_negatives

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        precision = float(self._true_positives) / (float(self._true_positives) + float(self._false_positives))
        recall = float(self._true_positives) / (float(self._true_positives) + float(self._false_negatives))
        accuracy = (float(self._true_positives) + float(self._true_negatives)) / float(self._total_count)
        f1_score = 2.0 * precision * recall / (precision + recall)
        if reset:
            self.reset()
        return precision, recall, accuracy, f1_score

    @overrides
    def reset(self):
        self._true_positives = 0.
        self._false_positives = 0.
        self._false_negatives = 0.
        self._true_negatives = 0.
        self._total_count = 0.
