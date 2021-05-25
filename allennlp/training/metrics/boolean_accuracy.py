from typing import Optional

from overrides import overrides
import torch

from allennlp.nn.util import dist_reduce_sum
from allennlp.training.metrics.metric import Metric


@Metric.register("boolean_accuracy")
class BooleanAccuracy(Metric):
    """
    Just checks batch-equality of two tensors and computes an accuracy metric based on that.
    That is, if your prediction has shape (batch_size, dim_1, ..., dim_n), this metric considers that
    as a set of `batch_size` predictions and checks that each is *entirely* correct across the remaining dims.
    This means the denominator in the accuracy computation is `batch_size`, with the caveat that predictions
    that are totally masked are ignored (in which case the denominator is the number of predictions that have
    at least one unmasked element).

    This is similar to [`CategoricalAccuracy`](./categorical_accuracy.md), if you've already done a `.max()`
    on your predictions.  If you have categorical output, though, you should typically just use
    `CategoricalAccuracy`.  The reason you might want to use this instead is if you've done
    some kind of constrained inference and don't have a prediction tensor that matches the API of
    `CategoricalAccuracy`, which assumes a final dimension of size `num_classes`.
    """

    def __init__(self) -> None:
        self._correct_count = 0.0
        self._total_count = 0.0

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        """
        # Parameters

        predictions : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, ...).
        gold_labels : `torch.Tensor`, required.
            A tensor of the same shape as `predictions`.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of the same shape as `predictions`.
        """
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)

        # Some sanity checks.
        if gold_labels.size() != predictions.size():
            raise ValueError(
                f"gold_labels must have shape == predictions.size() but "
                f"found tensor of shape: {gold_labels.size()}"
            )
        if mask is not None and mask.size() != predictions.size():
            raise ValueError(
                f"mask must have shape == predictions.size() but "
                f"found tensor of shape: {mask.size()}"
            )

        batch_size = predictions.size(0)

        if mask is not None:
            # We can multiply by the mask up front, because we're just checking equality below, and
            # this way everything that's masked will be equal.
            predictions = predictions * mask
            gold_labels = gold_labels * mask

            # We want to skip predictions that are completely masked;
            # so we'll keep predictions that aren't.
            keep = mask.view(batch_size, -1).max(dim=1)[0]
        else:
            keep = torch.ones(batch_size, device=predictions.device).bool()

        predictions = predictions.view(batch_size, -1)
        gold_labels = gold_labels.view(batch_size, -1)

        # At this point, predictions is (batch_size, rest_of_dims_combined),
        # so .eq -> .prod will be 1 if every element of the instance prediction is correct
        # and 0 if at least one element of the instance prediction is wrong.
        # Because of how we're handling masking, masked positions are automatically "correct".
        correct = predictions.eq(gold_labels).prod(dim=1).float()

        # Since masked positions are correct, we need to explicitly exclude instance predictions
        # where the entire prediction is masked (because they look "correct").
        _correct_count = (correct * keep).sum()
        _total_count = keep.sum()

        self._correct_count += dist_reduce_sum(_correct_count).item()
        self._total_count += dist_reduce_sum(_total_count).item()

    def get_metric(self, reset: bool = False):
        """
        # Returns

        The accumulated accuracy.
        """
        if self._total_count > 0:
            accuracy = float(self._correct_count) / float(self._total_count)
        else:
            accuracy = 0.0
        if reset:
            self.reset()
        return accuracy

    @overrides
    def reset(self):
        self._correct_count = 0.0
        self._total_count = 0.0
