from typing import Dict, Iterable, List, Optional, Tuple, Union, Any

import torch
import torch.distributed as dist

from allennlp.common.registrable import Registrable


class Metric(Registrable):
    """
    A very general abstract class representing a metric which can be
    accumulated.
    """

    def __call__(
        self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.BoolTensor]
    ):
        """
        # Parameters

        predictions : `torch.Tensor`, required.
            A tensor of predictions.
        gold_labels : `torch.Tensor`, required.
            A tensor corresponding to some gold label to evaluate against.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A mask can be passed, in order to deal with metrics which are
            computed over potentially padded elements, such as sequence labels.
        """
        raise NotImplementedError

    def get_metric(
        self, reset: bool, cuda_device: Union[int, torch.device] = torch.device("cpu"),
    ) -> Union[float, Tuple[float, ...], Dict[str, float], Dict[str, List[float]]]:
        """
        Compute and return the metric. Optionally also call `self.reset`.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        raise NotImplementedError

    @staticmethod
    def detach_tensors(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures the tensors are detached.
        """
        # Check if it's actually a tensor in case something else was passed.
        return (x.detach() if isinstance(x, torch.Tensor) else x for x in tensors)

    @staticmethod
    def _aggregate_metrics(
        metrics: Dict[str, Any],
        world_size: int = 1,
        cuda_device: Union[int, torch.device] = torch.device("cpu"),
    ) -> Dict[str, Any]:
        """
        Aggregate metrics across different processes. The default is to
        take the average.
        """
        # raise NotImplementedError
        aggregated_metrics = {}
        for metric_name, metric_val in metrics.items():
            metric_tensor = torch.tensor(metric_val).to(cuda_device)
            dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
            reduced_metric = metric_tensor.item() / world_size
            aggregated_metrics[metric_name] = reduced_metric
        return aggregated_metrics
