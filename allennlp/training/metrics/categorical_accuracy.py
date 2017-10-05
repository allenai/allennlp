from typing import Optional

from overrides import overrides
from torch.autograd import Variable

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


@Metric.register("categorical_accuracy")
class CategoricalAccuracy(Metric):
    """
    Categorical Top-K accuracy. Assumes integer labels, with
    each item to be classified having a single correct class.
    """
    def __init__(self, top_k: int = 1) -> None:
        self._top_k = top_k
        self.correct_count = 0.
        self.total_count = 0.

    def __call__(self,
                 predictions: Variable,
                 gold_labels: Variable,
                 mask: Optional[Variable] = None):
        """
        Parameters
        ----------
        predictions : ``Variable``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``Variable``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``Variable``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        # Get the data from the Variables.
        predictions_, gold_labels_, mask_ = self.unwrap_to_tensors(predictions, gold_labels, mask)

        # Some sanity checks.
        num_classes = predictions_.size(-1)
        if gold_labels_.dim() != predictions_.dim() - 1:
            raise ConfigurationError("gold_labels must have dimension == predictions.size() - 1 but "
                                     "found tensor of shape: {}".format(predictions_.size()))
        if (gold_labels_ >= num_classes).any():
            raise ConfigurationError("A gold label passed to Categorical Accuracy contains an id >= {}, "
                                     "the number of classes.".format(num_classes))

        # Top K indexes of the predictions (or fewer, if there aren't K of them)
        top_k = predictions_.topk(min(self._top_k, predictions_.shape[-1]), -1)[1]

        # This is of shape (batch_size, ..., top_k).
        correct = top_k.eq(gold_labels_.long().unsqueeze(-1)).float()

        if mask_ is not None:
            correct *= mask_.unsqueeze(-1)
            self.total_count += mask_.sum()
        else:
            self.total_count += gold_labels_.numel()
        self.correct_count += correct.sum()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        accuracy = float(self.correct_count) / float(self.total_count)
        if reset:
            self.reset()
        return accuracy

    @overrides
    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0
