import logging
from typing import Optional

from overrides import overrides
import torch

# import torch.distributed as dist

from allennlp.common.util import is_distributed
from allennlp.training.metrics.metric import Metric

logger = logging.getLogger(__name__)


@Metric.register("covariance")
class Covariance(Metric):
    """
    This `Metric` calculates the unbiased sample covariance between two tensors.
    Each element in the two tensors is assumed to be a different observation of the
    variable (i.e., the input tensors are implicitly flattened into vectors and the
    covariance is calculated between the vectors).

    This implementation is mostly modeled after the streaming_covariance function in Tensorflow. See:
    <https://github.com/tensorflow/tensorflow/blob/v1.10.1/tensorflow/contrib/metrics/python/ops/metric_ops.py#L3127>

    The following is copied from the Tensorflow documentation:

    The algorithm used for this online computation is described in
    <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online>.
    Specifically, the formula used to combine two sample comoments is
    `C_AB = C_A + C_B + (E[x_A] - E[x_B]) * (E[y_A] - E[y_B]) * n_A * n_B / n_AB`
    The comoment for a single batch of data is simply `sum((x - E[x]) * (y - E[y]))`, optionally masked.
    """

    def __init__(self) -> None:
        self._total_prediction_mean = 0.0
        self._total_label_mean = 0.0
        self._total_co_moment = 0.0
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

        # Flatten predictions, gold_labels, and mask. We calculate the covariance between
        # the vectors, since each element in the predictions and gold_labels tensor is assumed
        # to be a separate observation.
        predictions = predictions.view(-1)
        gold_labels = gold_labels.view(-1)

        if mask is not None:
            mask = mask.view(-1)
            predictions = predictions * mask
            gold_labels = gold_labels * mask
            num_batch_items = torch.sum(mask).item()
        else:
            num_batch_items = gold_labels.numel()

        # Note that self._total_count must be a float or int at all times
        # If it is a 1-dimension Tensor, the previous count will equal the updated_count.
        # The sampe applies for previous_total_prediction_mean and
        # previous_total_label_mean below -- we handle this in the code by
        # calling .item() judiciously.
        previous_count = self._total_count
        updated_count = previous_count + num_batch_items

        batch_mean_prediction = torch.sum(predictions) / num_batch_items

        delta_mean_prediction = (
            (batch_mean_prediction - self._total_prediction_mean) * num_batch_items
        ) / updated_count
        previous_total_prediction_mean = self._total_prediction_mean

        batch_mean_label = torch.sum(gold_labels) / num_batch_items
        delta_mean_label = (
            (batch_mean_label - self._total_label_mean) * num_batch_items
        ) / updated_count
        previous_total_label_mean = self._total_label_mean

        batch_coresiduals = (predictions - batch_mean_prediction) * (gold_labels - batch_mean_label)

        if mask is not None:
            batch_co_moment = torch.sum(batch_coresiduals * mask)
        else:
            batch_co_moment = torch.sum(batch_coresiduals)

        delta_co_moment = batch_co_moment + (
            previous_total_prediction_mean - batch_mean_prediction
        ) * (previous_total_label_mean - batch_mean_label) * (
            previous_count * num_batch_items / updated_count
        )

        # Due to the online computation of covariance, the following code can
        # still lead to nan values in later iterations. To be revisited.

        # if is_distributed():

        #     # Note: this gives an approximate aggregation of the covariance.
        #     device = gold_labels.device
        #     delta_mean_prediction = torch.tensor(delta_mean_prediction, device=device)
        #     delta_mean_label = torch.tensor(delta_mean_label, device=device)
        #     delta_co_moment = torch.tensor(delta_co_moment, device=device)
        #     _total_count = torch.tensor(updated_count, device=device)
        #     dist.all_reduce(delta_mean_prediction, op=dist.ReduceOp.SUM)
        #     dist.all_reduce(delta_mean_label, op=dist.ReduceOp.SUM)
        #     dist.all_reduce(delta_co_moment, op=dist.ReduceOp.SUM)
        #     dist.all_reduce(_total_count, op=dist.ReduceOp.SUM)
        #     updated_count = _total_count.item()

        self._total_prediction_mean += delta_mean_prediction.item()
        self._total_label_mean += delta_mean_label.item()
        self._total_co_moment += delta_co_moment.item()
        self._total_count = updated_count

    def get_metric(self, reset: bool = False):
        """
        # Returns

        The accumulated covariance.
        """
        if is_distributed():
            raise RuntimeError("Distributed aggregation for Covariance is currently not supported.")
        covariance = self._total_co_moment / (self._total_count - 1)
        if reset:
            self.reset()
        return covariance

    @overrides
    def reset(self):
        self._total_prediction_mean = 0.0
        self._total_label_mean = 0.0
        self._total_co_moment = 0.0
        self._total_count = 0.0
