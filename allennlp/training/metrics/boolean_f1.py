from typing import Optional

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric


@Metric.register("boolean_f1")
class BooleanF1(Metric):
    """
    Computes precision, recall, f1 score from boolean predictions and labels
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
        Precison, recall, accuracy, and F1 score
        """
        precision = float(self._true_positives) / (float(self._true_positives) + float(self._false_positives))
        recall = float(self._true_positives) / (float(self._true_positives) + float(self._false_negatives))
        accuracy = (float(self._true_positives) + float(self._true_negatives)) / float(self._total_count)
        if precision + recall > 0.:
            f1_score = 2.0 * precision * recall / (precision + recall)
        else:
            f1_score = 0.
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
