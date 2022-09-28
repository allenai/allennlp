from typing import List, Dict

from allennlp.common.util import nan_safe_tensor_divide
from allennlp.training.metrics.metric import Metric
from allennlp.training.metrics.fbeta_measure import FBetaMeasure


@Metric.register("fbeta_verbose")
class FBetaVerboseMeasure(FBetaMeasure):
    """Compute precision, recall, F-measure and support for each class.

    This is basically the same as `FBetaMeasure` (the super class)
    with two differences:
        - it always returns a dictionary of floats, while `FBetaMeasure`
          can return a dictionary of lists (one element for each class).
        - it always returns precision, recall and F-measure for each
          class and also three averaged values for each metric: micro,
          macro and weighted averages.

    The returned dictionary contains keys with the following format:
        <class>-precision : `float`
        <class>-recall : `float`
        <class>-fscore : `float`
        <avg>-precision : `float`
        <avg>-recall : `float`
        <avg>-fscore : `float`
    where <class> is the index (or the label if `index_to_label` is given)
    of each class; and <avg> is `micro`, `macro` and `weighted`, one for
    each kind of average.

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

    labels : `List[int]`, optional (default = `None`)
        The set of labels to include. Labels present in the data can be excluded,
        for example, to calculate a multi-class average ignoring a majority
        negative class. Labels not present in the data will result in 0
        components in a macro or weighted average.

    index_to_label : `Dict[int, str]`, optional (default = `None`)
        A dictionary mapping indices to the corresponding label.
        If this map is giving, the provided metrics include the label
        instead of the index for each class.
    """

    def __init__(
        self,
        beta: float = 1.0,
        labels: List[int] = None,
        index_to_label: Dict[int, str] = None,
    ) -> None:
        super().__init__(beta=beta, average=None, labels=labels)
        self._index_to_label = index_to_label

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
        of each class; and <avg> is `micro`, `macro` and `weighted`, one for
        each kind of average.
        """
        if self._true_positive_sum is None or self._pred_sum is None or self._true_sum is None:
            raise RuntimeError("You have never called this metric before.")

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
            label = str(c)
            if self._index_to_label:
                label = self._index_to_label[c]
            all_metrics[f"{label}-precision"] = p
            all_metrics[f"{label}-recall"] = r
            all_metrics[f"{label}-fscore"] = f

        # macro average
        all_metrics["macro-precision"] = precision.mean().item()
        all_metrics["macro-recall"] = recall.mean().item()
        all_metrics["macro-fscore"] = fscore.mean().item()

        # weighted average
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

        # micro average
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
