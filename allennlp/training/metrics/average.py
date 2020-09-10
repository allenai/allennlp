from overrides import overrides

import torch
import torch.distributed as dist

from allennlp.common.util import is_distributed
from allennlp.training.metrics.metric import Metric


@Metric.register("average")
class Average(Metric):
    """
    This [`Metric`](./metric.md) breaks with the typical `Metric` API and just stores values that were
    computed in some fashion outside of a `Metric`.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    `Metric` API.
    """

    def __init__(self) -> None:
        self._total_value = 0.0
        self._count = 0

    @overrides
    def __call__(self, value):
        """
        # Parameters

        value : `float`
            The value to average.
        """
        _total_value = list(self.detach_tensors(value))[0]
        _count = 1
        if is_distributed():
            device = torch.device("cuda" if dist.get_backend() == "nccl" else "cpu")
            count = torch.tensor(_count).to(device)
            total_value = torch.tensor(_total_value).to(device)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_value, op=dist.ReduceOp.SUM)
            _count = count.item()
            _total_value = total_value.item()
        self._count += _count
        self._total_value += _total_value

    @overrides
    def get_metric(self, reset: bool = False):
        """
        # Returns

        The average of all values that were passed to `__call__`.
        """

        average_value = self._total_value / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return average_value

    @overrides
    def reset(self):
        self._total_value = 0.0
        self._count = 0
