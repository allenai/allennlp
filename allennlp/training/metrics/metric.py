from typing import Dict, Iterable, List, Optional, Tuple, Union
import collections.abc
import torch

from allennlp.common.registrable import Registrable


class Metric(Registrable):
    """
    A very general abstract class representing a metric which can be
    accumulated.
    """

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        # Parameters

        predictions : `torch.Tensor`, required.
            A tensor of predictions.
        gold_labels : `torch.Tensor`, required.
            A tensor corresponding to some gold label to evaluate against.
        mask : `torch.Tensor`, optional (default = None).
            A mask can be passed, in order to deal with metrics which are
            computed over potentially padded elements, such as sequence labels.
        """
        raise NotImplementedError

    def get_metric(
        self, reset: bool
    ) -> Union[float, Tuple[float, ...], Dict[str, float], Dict[str, List[float]]]:
        """
        Compute and return the metric. Optionally also call `self.reset`.
        """
        raise NotImplementedError

    def get_metric_name_value_pairs(
        self, default_name: str, reset: bool = False
    ) -> Iterable[Tuple[str, float]]:
        """
        Return the metric as in `self.get_metric` but as an iterable of string-float pairs.
        """
        value = self.get_metric(reset)
        if isinstance(value, collections.abc.Mapping):
            for sub_name, sub_value in value.items():
                if isinstance(sub_value, collections.abc.Iterable):
                    for i, sub_value_i in enumerate(sub_value):
                        yield f"{sub_name}_{i}", sub_value_i
                else:
                    yield sub_name, sub_value
        elif isinstance(value, collections.abc.Iterable):
            for i, sub_value in enumerate(value):  # type: ignore
                yield f"{default_name}_{i}", sub_value  # type: ignore
        else:
            yield default_name, value

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        raise NotImplementedError

    @staticmethod
    def unwrap_to_tensors(*tensors: torch.Tensor):
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures that you're using tensors directly and that they are on
        the CPU.
        """
        return (x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in tensors)
