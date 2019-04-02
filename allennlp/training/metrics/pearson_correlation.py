from typing import Optional
import math

from overrides import overrides
import torch

from allennlp.training.metrics.covariance import Covariance
from allennlp.training.metrics.metric import Metric


@Metric.register("pearson_correlation")
class PearsonCorrelation(Metric):
    """
    This ``Metric`` calculates the sample Pearson correlation coefficient (r)
    between two tensors. Each element in the two tensors is assumed to be
    a different observation of the variable (i.e., the input tensors are
    implicitly flattened into vectors and the correlation is calculated
    between the vectors).

    This implementation is mostly modeled after the streaming_pearson_correlation function in Tensorflow. See
    https://github.com/tensorflow/tensorflow/blob/v1.10.1/tensorflow/contrib/metrics/python/ops/metric_ops.py#L3267

    This metric delegates to the Covariance metric the tracking of three [co]variances:

    - ``covariance(predictions, labels)``, i.e. covariance
    - ``covariance(predictions, predictions)``, i.e. variance of ``predictions``
    - ``covariance(labels, labels)``, i.e. variance of ``labels``

    If we have these values, the sample Pearson correlation coefficient is simply:

    r = covariance / (sqrt(predictions_variance) * sqrt(labels_variance))
    """
    def __init__(self) -> None:
        self._predictions_labels_covariance = Covariance()
        self._predictions_variance = Covariance()
        self._labels_variance = Covariance()

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
        self._predictions_labels_covariance(predictions, gold_labels, mask)
        self._predictions_variance(predictions, predictions, mask)
        self._labels_variance(gold_labels, gold_labels, mask)

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated sample Pearson correlation.
        """
        covariance = self._predictions_labels_covariance.get_metric(reset=reset)
        predictions_variance = self._predictions_variance.get_metric(reset=reset)
        labels_variance = self._labels_variance.get_metric(reset=reset)
        if reset:
            self.reset()
        pearson_r = covariance / (math.sqrt(predictions_variance) * math.sqrt(labels_variance))
        return pearson_r

    @overrides
    def reset(self):
        self._predictions_labels_covariance.reset()
        self._predictions_variance.reset()
        self._labels_variance.reset()
