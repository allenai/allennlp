# pylint: disable=no-self-use,invalid-name,protected-access
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.metrics import BooleanAccuracy


class BooleanAccuracyTest(AllenNlpTestCase):
    def test_accuracy_computation(self):
        accuracy = BooleanAccuracy()
        predictions = torch.Tensor([[0, 1],
                                    [2, 3],
                                    [4, 5],
                                    [6, 7]])
        targets = torch.Tensor([[0, 1],
                                [2, 2],
                                [4, 5],
                                [7, 7]])
        accuracy(predictions, targets)
        assert accuracy.get_metric() == 2. / 4

        mask = torch.ones(4, 2)
        mask[1, 1] = 0
        accuracy(predictions, targets, mask)
        assert accuracy.get_metric() == 5. / 8

        targets[1, 1] = 3
        accuracy(predictions, targets)
        assert accuracy.get_metric() == 8. / 12

        accuracy.reset()
        accuracy(predictions, targets)
        assert accuracy.get_metric() == 3. / 4
