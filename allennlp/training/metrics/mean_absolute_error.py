from typing import Optional, Union

from overrides import overrides
import torch
import torch.distributed as dist

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

        absolute_errors = torch.abs(predictions - gold_labels)
        if mask is not None:
            absolute_errors *= mask
            self._total_count += torch.sum(mask)
        else:
            self._total_count += gold_labels.numel()
        self._absolute_error += torch.sum(absolute_errors)

    def get_metric(
        self,
        reset: bool = False,
        world_size: int = 1,
        cuda_device: Union[int, torch.device] = torch.device("cpu"),
    ):
        """
        # Returns

        The accumulated mean absolute error.
        """
        mean_absolute_error = self._absolute_error / self._total_count
        if world_size > 1:
            mean_absolute_error_tensor = torch.tensor(mean_absolute_error).to(cuda_device)
            dist.all_reduce(mean_absolute_error_tensor, op=dist.ReduceOp.SUM)
            mean_absolute_error = mean_absolute_error_tensor.item() / world_size
        if reset:
            self.reset()
        return {"mae": mean_absolute_error}

    @overrides
    def reset(self):
        self._absolute_error = 0.0
        self._total_count = 0.0
