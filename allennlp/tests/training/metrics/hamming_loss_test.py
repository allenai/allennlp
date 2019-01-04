# pylint: disable=no-self-use,invalid-name,protected-access
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.metrics import HammingLoss


class HammingLossTest(AllenNlpTestCase):
    def test_hamming_loss(self):
        loss = HammingLoss()
        predictions = torch.Tensor([1, 2, 3, 4])
        targets = torch.Tensor([2, 2, 3, 4])
        loss(predictions, targets)
        assert loss.get_metric() == 1. / 4

        mask = torch.Tensor([1, 1, 1, 0])
        loss(predictions, targets, mask)
        assert loss.get_metric() == 2. / 7

        targets[2] = 2
        loss(predictions, targets)
        assert loss.get_metric() == 4. / 11

        loss.reset()
        loss(predictions, targets)
        assert loss.get_metric() == 2. / 4
