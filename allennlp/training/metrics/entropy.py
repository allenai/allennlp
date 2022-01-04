from typing import Optional


import torch

from allennlp.nn.util import dist_reduce_sum
from allennlp.training.metrics.metric import Metric


@Metric.register("entropy")
class Entropy(Metric):
    def __init__(self) -> None:
        self._entropy = 0.0
        self._count = 0

    def __call__(  # type: ignore
        self,
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

        _entropy = entropy.sum() / mask.sum()

        self._entropy += dist_reduce_sum(_entropy).item()
        self._count += dist_reduce_sum(1)

    def get_metric(self, reset: bool = False):
        """
        # Returns

        The scalar average entropy.
        """
        average_value = self._entropy / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return {"entropy": average_value}

    def reset(self):
        self._entropy = 0.0
        self._count = 0
