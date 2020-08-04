from typing import Optional, Union

from overrides import overrides
import torch
import torch.distributed as dist

from allennlp.common.util import is_distributed
from allennlp.training.metrics.metric import Metric


@Metric.register("entropy")
class Entropy(Metric):
    def __init__(self) -> None:
        self._entropy = 0.0
        self._count = 0

    @overrides
    def __call__(
        self,  # type: ignore
        logits: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        """
        # Parameters

        logits : `torch.Tensor`, required.
            A tensor of unnormalized log probabilities of shape (batch_size, ..., num_classes).
        mask : `torch.BoolTensor`, optional (default = `None`).
            A masking tensor of shape (batch_size, ...).
        """
        logits, mask = self.detach_tensors(logits, mask)

        if mask is None:
            mask = torch.ones(logits.size()[:-1], device=logits.device).bool()

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        probabilities = torch.exp(log_probs) * mask.unsqueeze(-1)
        weighted_negative_likelihood = -log_probs * probabilities
        entropy = weighted_negative_likelihood.sum(-1)

        self._entropy += entropy.sum() / mask.sum()
        self._count += 1

    @overrides
    def get_metric(
        self, reset: bool = False, cuda_device: Union[int, torch.device] = torch.device("cpu"),
    ):
        """
        # Returns

        The scalar average entropy.
        """
        if is_distributed():
            world_size = dist.get_world_size()
            _entropy = torch.tensor(self._entropy).to(cuda_device)
            _count = torch.tensor(self._count).to(cuda_device)
            dist.all_reduce(_entropy, op=dist.ReduceOp.SUM)
            dist.all_reduce(_count, op=dist.ReduceOp.SUM)
            self._entropy = _entropy.item() / world_size
            self._count = _count.item() / world_size
        average_value = self._entropy / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return {"entropy": average_value}

    @overrides
    def reset(self):
        self._entropy = 0.0
        self._count = 0
