from typing import Optional

from overrides import overrides
import torch
import scipy.stats as stats

from allennlp.training.metrics.metric import Metric


@Metric.register("spearman_correlation")
class SpearmanCorrelation(Metric):
    """
    This `Metric` calculates the sample Spearman correlation coefficient (r)
    between two tensors. Each element in the two tensors is assumed to be
    a different observation of the variable (i.e., the input tensors are
    implicitly flattened into vectors and the correlation is calculated
    between the vectors).

    <https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>
    """

    def __init__(self) -> None:
        super().__init__()
        self.total_predictions = torch.zeros(0)
        self.total_gold_labels = torch.zeros(0)

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
        # Flatten predictions, gold_labels, and mask. We calculate the Spearman correlation between
        # the vectors, since each element in the predictions and gold_labels tensor is assumed
        # to be a separate observation.
        predictions = predictions.reshape(-1)
        gold_labels = gold_labels.reshape(-1)

        self.total_predictions = self.total_predictions.to(predictions.device)
        self.total_gold_labels = self.total_gold_labels.to(gold_labels.device)

        if mask is not None:
            mask = mask.reshape(-1)
            self.total_predictions = torch.cat((self.total_predictions, predictions * mask), 0)
            self.total_gold_labels = torch.cat((self.total_gold_labels, gold_labels * mask), 0)
        else:
            self.total_predictions = torch.cat((self.total_predictions, predictions), 0)
            self.total_gold_labels = torch.cat((self.total_gold_labels, gold_labels), 0)

    @overrides
    def get_metric(self, reset: bool = False):
        """
        # Returns

        The accumulated sample Spearman correlation.
        """
        spearman_correlation = stats.spearmanr(
            self.total_predictions.cpu().numpy(), self.total_gold_labels.cpu().numpy()
        )

        if reset:
            self.reset()

        return spearman_correlation[0]

    @overrides
    def reset(self):
        self.total_predictions = torch.zeros(0)
        self.total_gold_labels = torch.zeros(0)
