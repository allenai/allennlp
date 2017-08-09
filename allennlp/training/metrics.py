from typing import Optional, Tuple, Union

from overrides import overrides
import torch

from allennlp.common.registrable import Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params


class Metric(Registrable):
    """
    A very general abstract class representing a metric which can be
    accumulated.
    """
    def __call__(self, predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor]):

        raise NotImplementedError

    def get_metric(self, reset: bool) -> Union[float, Tuple[float, ...]]:
        """
        Compute and return the metric. Optionally also call :func:`self.reset`.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params):
        metric_type = params.pop_choice("type", cls.list_available())
        return cls.by_name(metric_type)(**params.as_dict())  # type: ignore


@Metric.register("categorical_accuracy")
class CategoricalAccuracy(Metric):
    """
    Categorical Top-K accuracy. Assumes integer labels, with
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
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
                A masking tensor the same size as ``gold_labels``.
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
        """
        Returns
        -------
        The accumulated accuracy.
        """
        accuracy = 100. * float(self.correct_count) / float(self.total_count)
        if reset:
            self.reset()
        return accuracy

    @overrides
    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0


@Metric.register("f1")
class F1Measure(Metric):
    """
    Computes Precision, Recall and F1 with respect to a given ``positive_label`` label.
    For example, for a BIO tagging scheme, you would pass the classification index of
    the tag you are interested in, resulting in the Precision, Recall and F1 score being
    calculated for this tag only.
    """
    def __init__(self, positive_label: int) -> None:
        self._positive_label = positive_label
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
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        num_classes = predictions.size(-1)
        if (gold_labels >= num_classes).any():
            raise ConfigurationError("A gold label passed to F1Measure contains an id >= {}, "
                                     "the number of classes.".format(num_classes))
        if mask is None:
            mask = torch.ones(gold_labels.size())
        mask = mask.float()
        gold_labels = gold_labels.float()
        positive_label_mask = gold_labels.eq(self._positive_label).float()
        negative_label_mask = 1.0 - positive_label_mask

        argmax_predictions = predictions.topk(1, -1)[1].float().squeeze(-1)

        # True Negatives: correct non-positive predictions.
        correct_null_predictions = (argmax_predictions !=
                                    self._positive_label).float() * negative_label_mask
        self.true_negatives += (correct_null_predictions.float() * mask).sum()

        # True Positives: correct positively labeled predictions.
        correct_non_null_predictions = (argmax_predictions ==
                                        self._positive_label).float() * positive_label_mask
        self.true_positives += (correct_non_null_predictions * mask).sum()

        # False Negatives: incorrect negatively labeled predictions.
        incorrect_null_predictions = (argmax_predictions !=
                                      self._positive_label).float() * positive_label_mask
        self.false_negatives += (incorrect_null_predictions * mask).sum()

        # False Positives: incorrect positively labeled predictions
        incorrect_non_null_predictions = (argmax_predictions ==
                                          self._positive_label).float() * negative_label_mask
        self.false_positives += (incorrect_non_null_predictions * mask).sum()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        Returns a tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        recall = float(self.true_positives) / float(self.true_positives + self.false_negatives + 1e-13)
        precision = float(self.true_positives) / float(self.true_positives + self.false_positives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        if reset:
            self.reset()
        return precision, recall, f1_measure

    def reset(self):
        self.true_positives = 0.0
        self.true_negatives = 0.0
        self.false_positives = 0.0
        self.false_negatives = 0.0
        self.total_counts = 0.0
