from typing import Optional

from overrides import overrides
import torch
from sklearn import metrics

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


@Metric.register("auc")
class Auc(Metric):
    """
    The AUC Metric measures the area under the receiver-operating characteristic
    (ROC) curve for binary classification problems.
    """

    def __init__(self, positive_label=1):
        super().__init__()
        self._positive_label = positive_label
        self._all_predictions = torch.FloatTensor()
        self._all_gold_labels = torch.LongTensor()

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        """
        # Parameters

        predictions : `torch.Tensor`, required.
            A one-dimensional tensor of prediction scores of shape (batch_size).
        gold_labels : `torch.Tensor`, required.
            A one-dimensional label tensor of shape (batch_size), with {1, 0}
            entries for positive and negative class. If it's not binary,
            `positive_label` should be passed in the initialization.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A one-dimensional label tensor of shape (batch_size).
        """

        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)

        # Sanity checks.
        if gold_labels.dim() != 1:
            raise ConfigurationError(
                "gold_labels must be one-dimensional, "
                "but found tensor of shape: {}".format(gold_labels.size())
            )
        if predictions.dim() != 1:
            raise ConfigurationError(
                "predictions must be one-dimensional, "
                "but found tensor of shape: {}".format(predictions.size())
            )

        unique_gold_labels = torch.unique(gold_labels)
        if unique_gold_labels.numel() > 2:
            raise ConfigurationError(
                "AUC can be used for binary tasks only. gold_labels has {} unique labels, "
                "expected at maximum 2.".format(unique_gold_labels.numel())
            )

        gold_labels_is_binary = set(unique_gold_labels.tolist()) <= {0, 1}
        if not gold_labels_is_binary and self._positive_label not in unique_gold_labels:
            raise ConfigurationError(
                "gold_labels should be binary with 0 and 1 or initialized positive_label "
                "{} should be present in gold_labels".format(self._positive_label)
            )

        if mask is None:
            batch_size = gold_labels.shape[0]
            mask = torch.ones(batch_size, device=gold_labels.device).bool()

        self._all_predictions = self._all_predictions.to(predictions.device)
        self._all_gold_labels = self._all_gold_labels.to(gold_labels.device)

        self._all_predictions = torch.cat(
            [self._all_predictions, torch.masked_select(predictions, mask).float()], dim=0
        )
        self._all_gold_labels = torch.cat(
            [self._all_gold_labels, torch.masked_select(gold_labels, mask).long()], dim=0
        )

    def get_metric(self, reset: bool = False):
        if self._all_gold_labels.shape[0] == 0:
            return 0.5
        false_positive_rates, true_positive_rates, _ = metrics.roc_curve(
            self._all_gold_labels.cpu().numpy(),
            self._all_predictions.cpu().numpy(),
            pos_label=self._positive_label,
        )
        auc = metrics.auc(false_positive_rates, true_positive_rates)
        if reset:
            self.reset()
        return auc

    @overrides
    def reset(self):
        self._all_predictions = torch.FloatTensor()
        self._all_gold_labels = torch.LongTensor()
