from typing import Dict
import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.nn.initializers import Initializer, InitializerApplicator
from allennlp.common.params import Params

@Model.register("mlp")
class MultilayerPerceptron(Model):
    """
    This is just a thin wrapper around our ``FeedForward`` module.
    """
    def __init__(self, feed_forward: FeedForward) -> None:
        super().__init__()
        self.loss = torch.nn.MSELoss()
        self.module = feed_forward

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:  # pylint: disable=arguments-differ
        output = self.module.forward(x)
        loss = self.loss(output, y)

        return {"output": output, "loss": loss}

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'MultilayerPerceptron':
        feed_forward = FeedForward.from_params(params)
        return MultilayerPerceptron(feed_forward)
