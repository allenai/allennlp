from typing import Optional
import math
import torch

from overrides import overrides

from allennlp.training.metrics.entropy import Entropy
from allennlp.training.metrics.metric import Metric


@Metric.register("perplexity")
class Perplexity(Metric):
    def __init__(self) -> None:
        self._entropy = Entropy()

    @overrides
    def __call__(self,  # type: ignore
                 logits: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        self._entropy(logits, mask.float())

    @overrides
    def get_metric(self, reset: bool = False):
        entropy_val = self._entropy.get_metric(reset)
        ppl = math.exp(min(entropy_val, 100))
        return {"Perplexity": ppl}

    @overrides
    def reset(self):
        self._entropy.reset()