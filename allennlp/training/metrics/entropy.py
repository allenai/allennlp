from typing import Optional

from overrides import overrides
import torch
from torch.autograd import Variable

from allennlp.training.metrics.metric import Metric


@Metric.register("entropy")
class Entropy(Metric):

    def __init__(self) -> None:
        self._entropy = 0.0
        self._count = 0

    @overrides
    def __call__(self,  # type: ignore
                 logits: Variable,
                 mask: Optional[Variable] = None):
        """
        Parameters
        ----------
        logits : ``Variable``, required.
            A tensor of unnormalized log probabilities of shape (batch_size, ..., num_classes).
        mask: ``Variable``, optional (default = None).
            A masking tensor of shape (batch_size, ...).
        """
        # Get the data from the Variables.
        logits_, mask_ = self.unwrap_to_tensors(logits, mask)

        if mask_ is None:
            mask_ = torch.ones(*logits_.size()[:-1])

        log_probs = torch.nn.functional.log_softmax(logits_).data
        probabilities = torch.exp(log_probs) * mask_.unsqueeze(-1)
        weighted_negative_likelihood = - log_probs * probabilities
        entropy = weighted_negative_likelihood.sum(-1)

        self._entropy += entropy.sum() / mask_.sum()
        self._count += 1

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The scalar average entropy.
        """
        average_value = self._entropy / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return average_value

    @overrides
    def reset(self):
        self._entropy = 0.0
        self._count = 0
