import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model


class FakeModelForTestingNormalizationBiasVerification(Model):
    def __init__(self, use_bias=True):
        super().__init__(vocab=Vocabulary())
        self.conv = torch.nn.Conv2d(3, 5, kernel_size=1, bias=use_bias)
        self.bn = torch.nn.BatchNorm2d(5)

    def forward(self, x):
        # x: (B, 3, H, W)
        out = self.bn(self.conv(x))
        return {"loss": out.sum()}
