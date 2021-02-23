from overrides import overrides

from allennlp.training.metrics.metric import Metric
from allennlp.nn.util import dist_reduce, ReduceOp


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
        self._count += dist_reduce(1, ReduceOp.SUM)
        self._total_value += dist_reduce(float(list(self.detach_tensors(value))[0]), ReduceOp.SUM)

    @overrides
    def get_metric(self, reset: bool = False):
        """
        # Returns

        The average of all values that were passed to `__call__`.
        """

        average_value = self._total_value / self._count if self._count > 0 else 0.0
        if reset:
            self.reset()
        return float(average_value)

    @overrides
    def reset(self):
        self._total_value = 0.0
        self._count = 0
