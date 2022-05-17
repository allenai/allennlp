from typing import List, Optional, Union, Dict

import torch


from allennlp.common.util import nan_safe_tensor_divide
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric
from allennlp.training.metrics.fbeta_measure import FBetaMeasure
from allennlp.nn.util import dist_reduce_sum


@Metric.register("fbeta2")
class FBetaMeasure2(FBetaMeasure):
    """Compute precision, recall, F-measure and support for each class.

    This is basically the same as `FBetaMeasure` (the super class)
    with two differences:
        - it always returns a dictionary of floats, while `FBetaMeasure`
          can return a dictionary of lists (one element for each class).
        - it always returns precision, recall and F-measure for each
          class and also the averaged values (prefixed with `overall`).

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

    average : `Union[str, List[str]]`, optional (default = `None`)

        It can be one (or a list) of the following:

        `None` or `'micro'`:
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

    labels : `List[int]`, optional
        The set of labels to include. Labels present in the data can be excluded,
        for example, to calculate a multi-class average ignoring a majority
        negative class. Labels not present in the data will result in 0
        components in a macro or weighted average.

    index_to_label : `Dict[int, str]`, optional
        A dictionary mapping indices to the corresponding label.
        If this map is giving, the provided metrics include the label
        instead of the index for each class.
    """

    def __init__(
        self,
        beta: float = 1.0,
        average: Union[str, List[str]] = None,
        labels: List[int] = None,
        index_to_label: Dict[int, str] = None,
    ) -> None:
        super().__init__(beta=beta, average=None, labels=labels)
        self._index_to_label = index_to_label
        if not average:
            average = ["micro"]
        elif isinstance(average, str):
            average = [average]

        average_options = {"micro", "macro", "weighted"}
        if not all([a in average_options for a in average]):
            raise ConfigurationError(
                f"`average` has to be `None` or one (or a list) of {average_options}."
            )

        self._average = average

    def get_metric(self, reset: bool = False):
        """
        # Returns

        <class>-precision : `float`
        <class>-recall : `float`
        <class>-fscore : `float`
        <avg>-precision : `float`
        <avg>-recall : `float`
        <avg>-fscore : `float`

        where <class> is the index (or the label if `index_to_label` is given)
        of each class or its label (in case `index_to_label` is given);
        and <avg> is the option (or each one of the options) given in `average`.
        """
        if self._true_positive_sum is None:
            raise RuntimeError("You never call this metric before.")

        tp_sum = self._true_positive_sum
        pred_sum = self._pred_sum
        true_sum = self._true_sum

        if self._labels is not None:
            # Retain only selected labels and order them
            tp_sum = tp_sum[self._labels]
            pred_sum = pred_sum[self._labels]  # type: ignore
            true_sum = true_sum[self._labels]  # type: ignore

        beta2 = self._beta**2

        # Finally, we have all our sufficient statistics.
        precision = nan_safe_tensor_divide(tp_sum, pred_sum)
        recall = nan_safe_tensor_divide(tp_sum, true_sum)
        fscore = nan_safe_tensor_divide(
            (1 + beta2) * precision * recall, beta2 * precision + recall
        )

        all_metrics = {}
        for c, (p, r, f) in enumerate(zip(precision.tolist(), recall.tolist(), fscore.tolist())):
            if self._index_to_label:
                c = self._index_to_label[c]
            all_metrics[f"{c}-precision"] = p
            all_metrics[f"{c}-recall"] = r
            all_metrics[f"{c}-fscore"] = f

        if "macro" in self._average:
            all_metrics["macro-precision"] = precision.mean().item()
            all_metrics["macro-recall"] = recall.mean().item()
            all_metrics["macro-fscore"] = fscore.mean().item()

        if "weighted" in self._average:
            weights = true_sum
            weights_sum = true_sum.sum()  # type: ignore
            all_metrics["weighted-precision"] = nan_safe_tensor_divide(
                (weights * precision).sum(), weights_sum
            ).item()
            all_metrics["weighted-recall"] = nan_safe_tensor_divide(
                (weights * recall).sum(), weights_sum
            ).item()
            all_metrics["weighted-fscore"] = nan_safe_tensor_divide(
                (weights * fscore).sum(), weights_sum
            ).item()

        if "micro" in self._average:
            micro_precision = nan_safe_tensor_divide(tp_sum.sum(), pred_sum.sum())
            micro_recall = nan_safe_tensor_divide(tp_sum.sum(), true_sum.sum())
            all_metrics["micro-precision"] = micro_precision.item()
            all_metrics["micro-recall"] = micro_recall.item()
            all_metrics["micro-fscore"] = nan_safe_tensor_divide(
                (1 + beta2) * micro_precision * micro_recall, beta2 * micro_precision + micro_recall
            ).item()

        if reset:
            self.reset()

        return all_metrics
