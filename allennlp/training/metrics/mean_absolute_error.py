from typing import Dict, Optional

from overrides import overrides
import torch
import torch.distributed as dist

from allennlp.common.util import is_distributed
from allennlp.training.metrics.metric import Metric


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
        device = gold_labels.device

        absolute_errors = torch.abs(predictions - gold_labels)

        if mask is not None:
            absolute_errors *= mask
            _total_count = torch.sum(mask)
        else:
            _total_count = gold_labels.numel()
        _absolute_error = torch.sum(absolute_errors)

        if is_distributed():
            absolute_error = torch.tensor(_absolute_error, device=device)
            total_count = torch.tensor(_total_count, device=device)
            dist.all_reduce(absolute_error, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
            _absolute_error = absolute_error.item()
            _total_count = total_count.item()

        self._absolute_error += float(_absolute_error)
        self._total_count += int(_total_count)

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
