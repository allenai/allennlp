from typing import List, Optional, Union

import torch
import torch.distributed as dist
from overrides import overrides

from allennlp.common.util import is_distributed
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


@Metric.register("fbeta")
class FBetaMeasure(Metric):
    """Compute precision, recall, F-measure and support for each class.

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

    """

    def __init__(self, beta: float = 1.0, average: str = None, labels: List[int] = None) -> None:
        average_options = {None, "micro", "macro", "weighted"}
        if average not in average_options:
            raise ConfigurationError(f"`average` has to be one of {average_options}.")
        if beta <= 0:
            raise ConfigurationError("`beta` should be >0 in the F-beta score.")
        if labels is not None and len(labels) == 0:
            raise ConfigurationError("`labels` cannot be an empty list.")
        self._beta = beta
        self._average = average
        self._labels = labels

        # statistics
        # the total number of true positive instances under each class
        # Shape: (num_classes, )
        self._true_positive_sum: Union[None, torch.Tensor] = None
        # the total number of instances
        # Shape: (num_classes, )
        self._total_sum: Union[None, torch.Tensor] = None
        # the total number of instances under each _predicted_ class,
        # including true positives and false positives
        # Shape: (num_classes, )
        self._pred_sum: Union[None, torch.Tensor] = None
        # the total number of instances under each _true_ class,
        # including true positives and false negatives
        # Shape: (num_classes, )
        self._true_sum: Union[None, torch.Tensor] = None

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
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the `predictions` tensor without the `num_classes` dimension.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A masking tensor the same size as `gold_labels`.
        """
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)
        device = gold_labels.device

        # Calculate true_positive_sum, true_negative_sum, pred_sum, true_sum
        num_classes = predictions.size(-1)
        if (gold_labels >= num_classes).any():
            raise ConfigurationError(
                "A gold label passed to FBetaMeasure contains "
                f"an id >= {num_classes}, the number of classes."
            )

        # It means we call this metric at the first time
        # when `self._true_positive_sum` is None.
        if self._true_positive_sum is None:
            self._true_positive_sum = torch.zeros(num_classes, device=predictions.device)
            self._true_sum = torch.zeros(num_classes, device=predictions.device)
            self._pred_sum = torch.zeros(num_classes, device=predictions.device)
            self._total_sum = torch.zeros(num_classes, device=predictions.device)

        if mask is None:
            mask = torch.ones_like(gold_labels).bool()
        gold_labels = gold_labels.float()

        # If the prediction tensor is all zeros, the record is not classified to any of the labels.
        pred_mask = predictions.sum(dim=-1) != 0
        argmax_predictions = predictions.max(dim=-1)[1].float()

        true_positives = (gold_labels == argmax_predictions) & mask & pred_mask
        true_positives_bins = gold_labels[true_positives]

        # Watch it:
        # The total numbers of true positives under all _predicted_ classes are zeros.
        if true_positives_bins.shape[0] == 0:
            true_positive_sum = torch.zeros(num_classes, device=predictions.device)
        else:
            true_positive_sum = torch.bincount(
                true_positives_bins.long(), minlength=num_classes
            ).float()

        pred_bins = argmax_predictions[mask & pred_mask].long()
        # Watch it:
        # When the `mask` is all 0, we will get an _empty_ tensor.
        if pred_bins.shape[0] != 0:
            pred_sum = torch.bincount(pred_bins, minlength=num_classes).float()
        else:
            pred_sum = torch.zeros(num_classes, device=predictions.device)

        gold_labels_bins = gold_labels[mask].long()
        if gold_labels.shape[0] != 0:
            true_sum = torch.bincount(gold_labels_bins, minlength=num_classes).float()
        else:
            true_sum = torch.zeros(num_classes, device=predictions.device)

        self._total_sum += mask.sum().to(torch.float)

        if is_distributed():
            true_positive_sum = torch.tensor(true_positive_sum).to(device)
            pred_sum = torch.tensor(pred_sum).to(device)
            true_sum = torch.tensor(true_sum).to(device)
            dist.all_reduce(true_positive_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(pred_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(true_sum, op=dist.ReduceOp.SUM)

        self._true_positive_sum += true_positive_sum
        self._pred_sum += pred_sum
        self._true_sum += true_sum

    @overrides
    def get_metric(self, reset: bool = False):
        """
        # Returns

        precisions : `List[float]`
        recalls : `List[float]`
        f1-measures : `List[float]`

        !!! Note
            If `self.average` is not `None`, you will get `float` instead of `List[float]`.
        """
        if self._true_positive_sum is None:
            raise RuntimeError("You never call this metric before.")

        else:
            tp_sum = self._true_positive_sum
            pred_sum = self._pred_sum
            true_sum = self._true_sum

        if self._labels is not None:
            # Retain only selected labels and order them
            tp_sum = tp_sum[self._labels]
            pred_sum = pred_sum[self._labels]  # type: ignore
            true_sum = true_sum[self._labels]  # type: ignore

        if self._average == "micro":
            tp_sum = tp_sum.sum()
            pred_sum = pred_sum.sum()  # type: ignore
            true_sum = true_sum.sum()  # type: ignore

        beta2 = self._beta ** 2
        # Finally, we have all our sufficient statistics.
        precision = _prf_divide(tp_sum, pred_sum)
        recall = _prf_divide(tp_sum, true_sum)
        fscore = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        fscore[tp_sum == 0] = 0.0

        if self._average == "macro":
            precision = precision.mean()
            recall = recall.mean()
            fscore = fscore.mean()
        elif self._average == "weighted":
            weights = true_sum
            weights_sum = true_sum.sum()  # type: ignore
            precision = _prf_divide((weights * precision).sum(), weights_sum)
            recall = _prf_divide((weights * recall).sum(), weights_sum)
            fscore = _prf_divide((weights * fscore).sum(), weights_sum)

        if reset:
            self.reset()

        if self._average is None:
            return {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "fscore": fscore.tolist(),
            }
        else:
            return {"precision": precision.item(), "recall": recall.item(), "fscore": fscore.item()}

    @overrides
    def reset(self) -> None:
        self._true_positive_sum = None
        self._pred_sum = None
        self._true_sum = None
        self._total_sum = None

    @property
    def _true_negative_sum(self):
        if self._total_sum is None:
            return None
        else:
            true_negative_sum = (
                self._total_sum - self._pred_sum - self._true_sum + self._true_positive_sum
            )
            return true_negative_sum


def _prf_divide(numerator, denominator):
    """Performs division and handles divide-by-zero.

    On zero-division, sets the corresponding result elements to zero.
    """
    result = numerator / denominator
    mask = denominator == 0.0
    if not mask.any():
        return result

    # remove nan
    result[mask] = 0.0
    return result
