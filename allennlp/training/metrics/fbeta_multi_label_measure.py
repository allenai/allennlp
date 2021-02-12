from typing import List, Optional

import torch
import torch.distributed as dist
from overrides import overrides

from allennlp.common.util import is_distributed
from allennlp.training.metrics import FBetaMeasure
from allennlp.training.metrics.metric import Metric


@Metric.register("fbeta_multi_label")
class FBetaMultiLabelMeasure(FBetaMeasure):
    """Compute precision, recall, F-measure and support for multi-label classification.

    The precision is the ratio `tp / (tp + fp)` where `tp` is the number of
    true positives and `fp` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The recall is the ratio `tp / (tp + fn)` where `tp` is the number of
    true positives and `fn` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The F-beta score can be interpreted as a weighted harmonic mean of
    the precision and recall, where an F-beta score reaches its best
    value at 1 and worst score at 0.

    If we have precision and recall, the F-beta score is simply:
    `F-beta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)`

    The F-beta score weights recall more than precision by a factor of
    `beta`. `beta == 1.0` means recall and precision are equally important.

    The support is the number of occurrences of each class in `y_true`.

    # Parameters

    beta : `float`, optional (default = `1.0`)
        The strength of recall versus precision in the F-score.

    average : `str`, optional (default = `None`)
        If `None`, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        `'micro'`:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        `'macro'`:
            Calculate metrics for each label, and find their unweighted mean.
            This does not take label imbalance into account.
        `'weighted'`:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.

    labels: `list`, optional
        The set of labels to include and their order if `average is None`.
        Labels present in the data can be excluded, for example to calculate a
        multi-class average ignoring a majority negative class. Labels not present
        in the data will result in 0 components in a macro or weighted average.

    threshold: `float`, optional (default = `0.5`)
        Logits over this threshold will be considered predictions for the corresponding class.

    """

    def __init__(
        self,
        beta: float = 1.0,
        average: str = None,
        labels: List[int] = None,
        threshold: float = 0.5,
    ) -> None:
        super().__init__(beta, average, labels)
        self._threshold = threshold

    @overrides
    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        """
        # Parameters

        predictions : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : `torch.Tensor`, required.
            A tensor of boolean labels of shape (batch_size, ..., num_classes). It must be the same
            shape as the `predictions`.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A masking tensor the same size as `gold_labels`.
        """
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)
        device = gold_labels.device

        # Calculate true_positive_sum, true_negative_sum, pred_sum, true_sum
        num_classes = predictions.size(-1)

        # It means we call this metric at the first time
        # when `self._true_positive_sum` is None.
        if self._true_positive_sum is None:
            self._true_positive_sum = torch.zeros(num_classes, device=predictions.device)
            self._true_sum = torch.zeros(num_classes, device=predictions.device)
            self._pred_sum = torch.zeros(num_classes, device=predictions.device)
            self._total_sum = torch.zeros(num_classes, device=predictions.device)

        if mask is None:
            mask = torch.ones_like(gold_labels, dtype=torch.bool)
        gold_labels = gold_labels.float()

        # If the prediction tensor is all zeros, the record is not classified to any of the labels.
        pred_mask = (predictions.sum(dim=-1) != 0).unsqueeze(-1)
        threshold_predictions = (predictions >= self._threshold).float()

        class_indices = (
            torch.arange(num_classes, device=predictions.device)
            .unsqueeze(0)
            .repeat(gold_labels.size(0), 1)
        )
        true_positives = (gold_labels * threshold_predictions).bool() & mask & pred_mask
        true_positives_bins = class_indices[true_positives]

        # Watch it:
        # The total numbers of true positives under all _predicted_ classes are zeros.
        if true_positives_bins.shape[0] == 0:
            true_positive_sum = torch.zeros(num_classes, device=predictions.device)
        else:
            true_positive_sum = torch.bincount(
                true_positives_bins.long(), minlength=num_classes
            ).float()

        pred_bins = class_indices[threshold_predictions.bool() & mask & pred_mask]
        # Watch it:
        # When the `mask` is all 0, we will get an _empty_ tensor.
        if pred_bins.shape[0] != 0:
            pred_sum = torch.bincount(pred_bins, minlength=num_classes).float()
        else:
            pred_sum = torch.zeros(num_classes, device=predictions.device)

        gold_labels_bins = class_indices[gold_labels.bool() & mask]
        if gold_labels_bins.shape[0] != 0:
            true_sum = torch.bincount(gold_labels_bins, minlength=num_classes).float()
        else:
            true_sum = torch.zeros(num_classes, device=predictions.device)

        self._total_sum += mask.expand_as(gold_labels).sum().to(torch.float)

        if is_distributed():
            true_positive_sum = torch.tensor(true_positive_sum, device=device)
            pred_sum = torch.tensor(pred_sum, device=device)
            true_sum = torch.tensor(true_sum, device=device)
            dist.all_reduce(true_positive_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(pred_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(true_sum, op=dist.ReduceOp.SUM)

        self._true_positive_sum += true_positive_sum
        self._pred_sum += pred_sum
        self._true_sum += true_sum

    @property
    def _true_negative_sum(self):
        if self._total_sum is None:
            return None
        else:
            true_negative_sum = (
                self._total_sum[0] / self._true_positive_sum.size(0)
                - self._pred_sum
                - self._true_sum
                + self._true_positive_sum
            )
            return true_negative_sum


@Metric.register("f1_multi_label")
class F1MultiLabelMeasure(FBetaMultiLabelMeasure):
    def __init__(
        self, average: str = None, labels: List[int] = None, threshold: float = 0.5
    ) -> None:
        super().__init__(1.0, average, labels, threshold)
