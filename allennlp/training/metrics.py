from typing import Dict
import torch
from allennlp.common.registrable import Registrable
from allennlp.common.checks import ConfigurationError

class Metric(Registrable):
    """
    A very general abstract class representing a metric which can be
    accumulated.
    """
    def __call__(self, *args, **kwargs):

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
    def __init__(self, top_k: int = 1):
        self.top_k = top_k
        self.correct_count = 0.
        self.total_count = 0.

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: torch.Tensor = None):
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
        batch_size, *_, num_classes = predictions.size()
        if gold_labels.dim() != predictions.dim() - 1:
            raise ConfigurationError("gold_labels must have dimension == predictions.size() - 1 but "
                                     "found tensor of shape: {}".format(predictions.size()))
        if (gold_labels >= num_classes).any():
            raise ConfigurationError("A gold label passed to Categorical Accuracy contains an id >= {}, "
                                     "the number of classes.".format(num_classes))

        # Top K indexes of the predictions.
        top_k = predictions.topk(self.top_k, -1)[1]

        correct = top_k.eq(gold_labels.long().unsqueeze(-1)).float()
        count = torch.ones(gold_labels.size())
        if mask:
            correct *= mask
            count *= mask

        self.correct_count += correct.sum()
        self.total_count += count.sum()

    def get_metric(self, reset: bool = False):
        accuracy = 100. * float(self.correct_count) / float(self.total_count)
        if reset:
            self.reset()
        return {"accuracy": accuracy}

    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0


@Metric.register("f1")
class F1Measure(Metric):

    def __init__(self, no_prediction_label: int):

        self._no_prediction_label = no_prediction_label
        self.true_positives = 0.0
        self.true_negatives = 0.0
        self.false_positives = 0.0
        self.false_negatives = 0.0
        self.total_counts = 0.0

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor):

        no_prediction_mask = gold_labels.eq(gold_labels)






