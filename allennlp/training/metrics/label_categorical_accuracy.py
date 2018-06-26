from typing import Optional

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy


@Metric.register("label_categorical_accuracy")
class LabelCategoricalAccuracy(CategoricalAccuracy):
    """
    Categorical Top-K accuracy for a specific label. Assumes
    integer labels, with each item to be classified having a
    single correct class.
    """
    def __init__(self, positive_label: int, top_k: int = 1) -> None:
        super(LabelCategoricalAccuracy, self).__init__(top_k)
        self._positive_label = positive_label

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

        # Some sanity checks.
        num_classes = predictions.size(-1)
        if gold_labels.dim() != predictions.dim() - 1:
            raise ConfigurationError("gold_labels must have dimension == predictions.size() - 1 but "
                                     "found tensor of shape: {}".format(predictions.size()))
        if (gold_labels >= num_classes).any():
            raise ConfigurationError("A gold label passed to Categorical Accuracy contains an id >= {}, "
                                     "the number of classes.".format(num_classes))

        # Top K indexes of the predictions (or fewer, if there aren't K of them).
        # Special case topk == 1, because it's common and .max() is much faster than .topk().
        if self._top_k == 1:
            top_k = predictions.max(-1)[1].unsqueeze(-1)
        else:
            top_k = predictions.topk(min(self._top_k, predictions.shape[-1]), -1)[1]

        # This is of shape (batch_size, ..., top_k).
        correct = top_k.eq(gold_labels.long().unsqueeze(-1)).float()

        if mask is None:
            mask = torch.ones_like(gold_labels)
        label_mask = gold_labels.eq(self._positive_label)
        # elementwise-AND between label_mask and mask to get a mask
        # over elements that are both 1
        mask = label_mask.long() & mask.long()

        correct *= mask.float().unsqueeze(-1)
        self.total_count += mask.sum()
        self.correct_count += correct.sum()
