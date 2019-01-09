from typing import Dict, List, Optional, Union

import torch
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


@Metric.register("fbeta")
class FBetaMeasure(Metric):
    """Compute precision, recall, F-measure and support for each class.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The F-beta score can be interpreted as a weighted harmonic mean of
    the precision and recall, where an F-beta score reaches its best
    value at 1 and worst score at 0.

    The F-beta score weights recall more than precision by a factor of
    ``beta``. ``beta == 1.0`` means recall and precision are equally important.

    The support is the number of occurrences of each class in ``y_true``.

    Parameters
    ----------
    beta : ``float``, optional (default = 1.0)
        The strength of recall versus precision in the F-score.

    average : string, [None (default), 'micro', 'macro']
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted mean.
            This does not take label imbalance into account.

    labels: list, optional
        The set of labels to include and their order if ``average is None``.
        Labels present in the data can be excluded, for example to calculate a
        multi-classes average ignoring a majority negative class, while labels not present
        in the data will result in 0 components in a macro average. For multi-labels
        targets, labels are column indices.

    """
    def __init__(self,
                 beta: float = 1.0,
                 average: str = None,
                 labels: List[int] = None) -> None:
        average_options = (None, 'micro', 'macro')
        if average not in average_options and average != 'binary':
            raise ConfigurationError(f"`average` has to be one of {average_options}.")
        if beta <= 0:
            raise ConfigurationError("`beta` should be >0 in the F-beta score.")

        self._beta = beta
        self._average = average
        self._labels = labels

        # statistics
        self._tp_sum = None
        self._pred_sum = None
        self._true_sum = None

    @overrides
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        # Calculate tp_sum, pred_sum, true_sum
        num_classes = predictions.size(-1)
        if (gold_labels >= num_classes).any():
            raise ConfigurationError("A gold label passed to FBetaMeasure contains "
                                     f"an id >= {num_classes}, the number of classes.")

        if self._tp_sum is None:
            self._tp_sum = torch.zeros(num_classes)
            self._true_sum = torch.zeros(num_classes)
            self._pred_sum = torch.zeros(num_classes)

        if mask is None:
            mask = torch.ones_like(gold_labels)
        mask = mask.to(torch.uint8)
        gold_labels = gold_labels.float()

        argmax_predictions = predictions.max(dim=-1)[1].float().squeeze(dim=-1)
        true_positives = (gold_labels == argmax_predictions) * mask
        true_positives_bins = gold_labels[true_positives]

        if true_positives_bins.shape[0] == 0:
            tp_sum = torch.zeros(num_classes)
        else:
            tp_sum = torch.bincount(true_positives_bins.long(), minlength=num_classes).float()

        pred_bins = argmax_predictions[mask].long()
        if pred_bins.shape[0] != 0:
            pred_sum = torch.bincount(pred_bins, minlength=num_classes).float()
        else:
            pred_sum = torch.zeros(num_classes)

        gold_labels_bins = gold_labels[mask].long()
        if gold_labels.shape[0] != 0:
            true_sum = torch.bincount(gold_labels_bins, minlength=num_classes).float()
        else:
            true_sum = torch.zeros(num_classes)

        self._tp_sum += tp_sum
        self._pred_sum += pred_sum
        self._true_sum += true_sum

    @overrides
    def get_metric(self,
                   reset: bool = False) -> Dict[str, Union[float, List[float]]]:
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precisions : List[float]
        recalls : List[float]
        f1-measures : List[float]

        If ``self.average`` is not None, you will get ``float`` instead of ``List[float]``.
        """
        if self._tp_sum is None:
            raise RuntimeError("You never call this metric before.")

        tp_sum = self._tp_sum
        pred_sum = self._pred_sum
        true_sum = self._true_sum

        if self._average == 'micro':
            tp_sum = tp_sum.sum()
            pred_sum = pred_sum.sum()
            true_sum = true_sum.sum()

        beta2 = self._beta ** 2
        # Finally, we have all our sufficient statistics.
        precision = _prf_divide(tp_sum, pred_sum)
        recall = _prf_divide(tp_sum, true_sum)
        fscore = ((1 + beta2) * precision * recall /
                  (beta2 * precision + recall))
        fscore[tp_sum == 0] = 0.0

        if self._average == 'macro':
            precision = precision.mean()
            recall = recall.mean()
            fscore = fscore.mean()

        if reset:
            self.reset()

        if self._labels is not None:
            # Retain only selected labels and order them
            precision = precision[self._labels]
            recall = recall[self._labels]
            fscore = fscore[self._labels]

        if self._average is None:
            return {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "fscore": fscore.tolist()
            }
        else:
            return {
                    "precision": precision.item(),
                    "recall": recall.item(),
                    "fscore": fscore.item()
            }

    @overrides
    def reset(self) -> None:
        self._tp_sum = None
        self._pred_sum = None
        self._true_sum = None


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
