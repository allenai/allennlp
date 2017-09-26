from typing import Dict, Optional, Tuple, Union
import torch

from allennlp.common.registrable import Registrable
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary


class Metric(Registrable):
    """
    A very general abstract class representing a metric which can be
    accumulated.
    """
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor]):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions.
        gold_labels : ``torch.Tensor``, required.
            A tensor corresponding to some gold label to evaluate against.
        mask: ``torch.Tensor``, optional (default = None).
            A mask can be passed, in order to deal with metrics which are
            computed over potentially padded elements, such as sequence labels.
        """
        raise NotImplementedError

    def get_metric(self, reset: bool) -> Union[float, Tuple[float, ...], Dict[str, float]]:
        """
        Compute and return the metric. Optionally also call :func:`self.reset`.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params, vocab: Optional[Vocabulary] = None):
        metric_type = params.pop_choice("type", cls.list_available())
        if vocab:
            params["vocabulary"] = vocab
        return cls.by_name(metric_type)(**params.as_dict())  # type: ignore

    @staticmethod
    def unwrap_to_tensors(*tensors):
        """
        If you actually passed in Variables to a Metric instead of Tensors, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures that you're using tensors directly and that they are on
        the CPU.
        """
        return (x.data.cpu() if isinstance(x, torch.autograd.Variable) else x for x in tensors)
