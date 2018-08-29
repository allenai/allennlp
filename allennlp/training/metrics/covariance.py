from typing import Optional

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric


@Metric.register("covariance")
class Covariance(Metric):
    """
    This ``Metric`` calculates the unbiased sample covariance between two tensors.
    """
    def __init__(self) -> None:
        self._total_prediction_mean = 0.0
        self._total_label_mean = 0.0
        self._total_comoment = 0.0
        self._total_count = 0.0

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ...).
        gold_labels : ``torch.Tensor``, required.
            A tensor of the same shape as ``predictions``.
        mask: ``torch.Tensor``, optional (default = None).
            A tensor of the same shape as ``predictions``.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)
        # Flatten predictions, gold_labels, and mask. We calculate the covariance between
        # the vectors, since each element in the predictions and gold_labels tensor is assumed
        # to be a separate observation.
        predictions = predictions.view(-1)
        gold_labels = gold_labels.view(-1)

        if mask is not None:
            mask = mask.view(-1)
            predictions *= mask
            gold_labels *= mask
            num_batch_items = torch.sum(mask).item()
        else:
            num_batch_items = gold_labels.numel()

        # Note that self._total_count must be a float or int at all times
        # If it is a 1-dimension Tensor, the previous count will equal the updated_count.
        # The sampe applies for previous_total_prediction_mean and
        # previous_total_label_mean below -- we handle this in the code by
        # calling .item() judiciously.
        previous_count = self._total_count
        updated_count = self._total_count + num_batch_items

        batch_mean_prediction = torch.sum(predictions) / num_batch_items
        delta_mean_prediction = ((batch_mean_prediction - self._total_prediction_mean) * num_batch_items) / updated_count
        previous_total_prediction_mean = self._total_prediction_mean
        self._total_prediction_mean += delta_mean_prediction.item()

        batch_mean_label = torch.sum(gold_labels) / num_batch_items
        delta_mean_label = ((batch_mean_label - self._total_label_mean) * num_batch_items) / updated_count
        previous_total_label_mean = self._total_label_mean
        self._total_label_mean += delta_mean_label.item()

        batch_coresiduals = (predictions - batch_mean_prediction) * (gold_labels - batch_mean_label)
        if mask is not None:
            batch_comoment = torch.sum(batch_coresiduals * mask)
        else:
            batch_comoment = torch.sum(batch_coresiduals)
        delta_comoment = (
            batch_comoment + (previous_total_prediction_mean - batch_mean_prediction) *
            (previous_total_label_mean - batch_mean_label) *
            (previous_count * num_batch_items / updated_count))
        self._total_comoment += delta_comoment.item()
        self._total_count = updated_count

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated covariance.
        """
        covariance = self._total_comoment / (self._total_count - 1)
        if reset:
            self.reset()
        return covariance

    @overrides
    def reset(self):
        self._total_prediction_mean = 0.0
        self._total_label_mean = 0.0
        self._total_comoment = 0.0
        self._total_count = 0.0
