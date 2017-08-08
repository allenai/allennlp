from typing import Dict, Optional

from overrides import overrides
import torch

from allennlp.common.registrable import Registrable
from allennlp.common.checks import ConfigurationError


class Metric(Registrable):
    """
    A very general abstract class representing a metric which can be
    accumulated.
    """
    def __call__(self, predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor]):

        raise NotImplementedError

    def get_metric(self, reset: bool) -> Dict[str, float]:
        """
        Compute and return the metric. Optionally also call :func:`self.reset`.
        """

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        raise NotImplementedError


@Metric.register("categorical_accuracy")
class CategoricalAccuracy(Metric):
    """
    Categorical TopK accuracy. Assumes integer labels, with
    each item to be classified having a single correct class.
    """
    def __init__(self, top_k: int = 1) -> None:
        self.top_k = top_k
        self.correct_count = 0.
        self.total_count = 0.

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size).
        mask: ``torch.Tensor``, optional (default = None).

        """
        # Some sanity checks.
        num_classes = predictions.size(-1)
        if gold_labels.dim() != predictions.dim() - 1:
            raise ConfigurationError("gold_labels must have dimension == predictions.size() - 1 but "
                                     "found tensor of shape: {}".format(predictions.size()))
        if (gold_labels >= num_classes).any():
            raise ConfigurationError("A gold label passed to Categorical Accuracy contains an id >= {}, "
                                     "the number of classes.".format(num_classes))

        # Top K indexes of the predictions.
        top_k = predictions.topk(self.top_k, -1)[1]

        # This is of shape (batch_size, ..., top_k).
        correct = top_k.eq(gold_labels.long().unsqueeze(-1)).float()
        count = torch.ones(gold_labels.size())
        if mask is not None:
            correct *= mask.unsqueeze(-1)
            count *= mask
        self.correct_count += correct.sum()
        self.total_count += count.sum()

    def get_metric(self, reset: bool = False):
        accuracy = 100. * float(self.correct_count) / float(self.total_count)
        if reset:
            self.reset()
        return {"accuracy": accuracy}

    @overrides
    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0


@Metric.register("f1")
class F1Measure(Metric):

    def __init__(self, null_prediction_label: int) -> None:
        self._null_prediction_label = null_prediction_label
        self.true_positives = 0.0
        self.true_negatives = 0.0
        self.false_positives = 0.0
        self.false_negatives = 0.0
        self.total_counts = 0.0

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size).
        mask: ``torch.Tensor``, optional (default = None).

        """
        if mask is None:
            mask = torch.ones(gold_labels.size())
        mask = mask.float()

        null_prediction_mask = gold_labels.eq(self._null_prediction_label).float()
        some_prediction_mask = 1.0 - null_prediction_mask

        argmax_predictions = predictions.topk(1, -1)[1].float().squeeze(-1)

        # True Negatives: correct null_prediction predictions.
        correct_null_predictions = (argmax_predictions ==
                                    self._null_prediction_label).float() * null_prediction_mask
        self.true_negatives += (correct_null_predictions.float() * mask).sum()

        # True Positives: correct non-null predictions.
        correct_non_null_predictions = (argmax_predictions ==
                                        gold_labels).float() * some_prediction_mask
        self.true_positives += (correct_non_null_predictions * mask).sum()

        # False Negatives: incorrect null_prediction predictions.
        incorrect_null_predictions = (argmax_predictions !=
                                      self._null_prediction_label).float() * null_prediction_mask
        self.false_negatives += (incorrect_null_predictions * mask).sum()

        # False Positives: incorrect non-null predictions
        incorrect_non_null_predictions = (argmax_predictions !=
                                          gold_labels).float() * some_prediction_mask
        self.false_positives += (incorrect_non_null_predictions * mask).sum()

    def get_metric(self, reset: bool = False):
        recall = float(self.true_positives) / float(self.true_positives + self.false_negatives)
        precision = float(self.true_positives) / float(self.true_positives + self.false_positives)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))

        if reset:
            self.reset()
        return {"precision": precision, "recall": recall, "f1-measure": f1_measure}

    def reset(self):
        self.true_positives = 0.0
        self.true_negatives = 0.0
        self.false_positives = 0.0
        self.false_negatives = 0.0
        self.total_counts = 0.0
