from typing import Dict, Tuple

from allennlp.training.metrics.metric import Metric
from allennlp.training.metrics.fbeta_measure import FBetaMeasure


@Metric.register("f1")
class F1Measure(FBetaMeasure):
    """
    Computes Precision, Recall and F1 with respect to a given ``positive_label``.
    For example, for a BIO tagging scheme, you would pass the classification index of
    the tag you are interested in, resulting in the Precision, Recall and F1 score being
    calculated for this tag only.
    """
    def __init__(self, positive_label: int) -> None:
        super().__init__(beta=1,
                         labels=[positive_label])

    def get_metric(self,
                   reset: bool = False) -> Tuple[float, float, float]:
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        metric = super().get_metric(reset=reset)
        return metric['precision'], metric['recall'], metric['fscore']

    @property
    def _true_positives(self):
        return self._tp_sum[0] if self._tp_sum is not None else 0.0

    @property
    def _true_negatives(self):
        raise DeprecationWarning("`_true_negatives` is not supported.")

    @property
    def _false_positives(self):
        return ((self._pred_sum[0] - self._true_positives) if self._pred_sum is not None
                else 0.0)

    @property
    def _false_negatives(self):
        return ((self._true_sum[0] - self._true_positives) if self._true_sum is not None
                else 0.0)
