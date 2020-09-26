from typing import Dict

from allennlp.training.metrics.metric import Metric
from allennlp.training.metrics.fbeta_measure import FBetaMeasure


@Metric.register("f1")
class F1Measure(FBetaMeasure):
    """
    Computes Precision, Recall and F1 with respect to a given `positive_label`.
    For example, for a BIO tagging scheme, you would pass the classification index of
    the tag you are interested in, resulting in the Precision, Recall and F1 score being
    calculated for this tag only.
    """

    def __init__(self, positive_label: int) -> None:
        super().__init__(beta=1, labels=[positive_label])
        self._positive_label = positive_label

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        """
        # Returns

        precision : `float`
        recall : `float`
        f1-measure : `float`
        """
        metric = super().get_metric(reset=reset)
        # Because we just care about the class `positive_label`
        # there is just one item in `precision`, `recall`, `fscore`
        precision = metric["precision"][0]
        recall = metric["recall"][0]
        f1 = metric["fscore"][0]
        return {"precision": precision, "recall": recall, "f1": f1}

    @property
    def _true_positives(self):
        # When this metric is never called, `self._true_positive_sum` is None,
        # under which case we return 0.0 for backward compatibility.
        if self._true_positive_sum is None:
            return 0.0
        else:
            return self._true_positive_sum[self._positive_label]

    @property
    def _true_negatives(self):
        # When this metric is never called, `self._true_negative_sum` is None,
        # under which case we return 0.0 for backward compatibility.
        if self._true_negative_sum is None:
            return 0.0
        else:
            return self._true_negative_sum[self._positive_label]

    @property
    def _false_positives(self):
        # When this metric is never called, `self._pred_sum` is None,
        # under which case we return 0.0 for backward compatibility.
        if self._pred_sum is None:
            return 0.0
        else:
            # `self._pred_sum` is the total number of instances under each _predicted_ class,
            # including true positives and false positives.
            return self._pred_sum[self._positive_label] - self._true_positives

    @property
    def _false_negatives(self):
        # When this metric is never called, `self._true_sum` is None,
        # under which case we return 0.0 for backward compatibility.
        if self._true_sum is None:
            return 0.0
        else:
            # `self._true_sum` is the total number of instances under each _true_ class,
            # including true positives and false negatives.
            return self._true_sum[self._positive_label] - self._true_positives
