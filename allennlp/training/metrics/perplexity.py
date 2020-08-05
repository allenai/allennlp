from typing import Dict
from overrides import overrides
import torch

from allennlp.training.metrics.average import Average
from allennlp.training.metrics.metric import Metric


@Metric.register("perplexity")
class Perplexity(Average):
    """
    Perplexity is a common metric used for evaluating how well a language model
    predicts a sample.

    Notes
    -----
    Assumes negative log likelihood loss of each batch (base e). Provides the
    average perplexity of the batches.
    """

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        """
        # Returns

        The accumulated perplexity.
        """
        average_loss = super().get_metric(reset)["average_value"]
        if average_loss == 0:
            perplexity = 0.0
        perplexity = float(torch.exp(average_loss))

        # Exponentiate the loss to compute perplexity
        return {"perplexity": perplexity}
