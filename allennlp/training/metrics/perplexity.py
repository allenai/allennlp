from overrides import overrides
import math

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
    def get_metric(self, reset: bool = False):
        """
        # Returns

        The accumulated perplexity.
        """
        average_loss = super().get_metric(reset)
        if average_loss == 0:
            return 0.0

        # Exponentiate the loss to compute perplexity
        return math.exp(average_loss)
