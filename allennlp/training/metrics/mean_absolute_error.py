from typing import Dict, Optional

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric
from allennlp.nn.util import dist_reduce_sum


@Metric.register("mean_absolute_error")
class MeanAbsoluteError(Metric):
    """
    This `Metric` calculates the mean absolute error (MAE) between two tensors.
    """

    def __init__(self) -> None:
        self._absolute_error = 0.0
        self._total_count = 0.0

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> None:
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

        absolute_errors = torch.abs(predictions - gold_labels)

        if mask is not None:
            absolute_errors *= mask
            _total_count = torch.sum(mask)
        else:
            _total_count = gold_labels.numel()
        _absolute_error = torch.sum(absolute_errors)

        self._absolute_error += float(dist_reduce_sum(_absolute_error))
        self._total_count += int(dist_reduce_sum(_total_count))

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        """
        # Returns

        The accumulated mean absolute error.
        """
        mean_absolute_error = self._absolute_error / self._total_count
        if reset:
            self.reset()
        return {"mae": mean_absolute_error}

    @overrides
    def reset(self) -> None:
        self._absolute_error = 0.0
        self._total_count = 0.0
