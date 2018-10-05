# pylint: disable=no-self-use,invalid-name,protected-access
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.metrics import MeanAbsoluteError


class MeanAbsoluteErrorTest(AllenNlpTestCase):
    def test_mean_absolute_error_computation(self):
        mae = MeanAbsoluteError()
        predictions = torch.Tensor([[1.0, 1.5, 1.0],
                                    [2.0, 3.0, 3.5],
                                    [4.0, 5.0, 5.5],
                                    [6.0, 7.0, 7.5]])
        targets = torch.Tensor([[0.0, 1.0, 0.0],
                                [2.0, 2.0, 0.0],
                                [4.0, 5.0, 0.0],
                                [7.0, 7.0, 0.0]])
        mae(predictions, targets)
        assert mae.get_metric() == 21.0 / 12.0

        mask = torch.Tensor([[1.0, 1.0, 0.0],
                             [1.0, 1.0, 0.0],
                             [1.0, 1.0, 0.0],
                             [1.0, 1.0, 0.0]])
        mae(predictions, targets, mask)
        assert mae.get_metric() == (21.0 + 3.5) / (12.0 + 8.0)

        new_targets = torch.Tensor([[2.0, 2.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [7.0, 7.0, 0.0],
                                    [4.0, 5.0, 0.0]])
        mae(predictions, new_targets)
        assert mae.get_metric() == (21.0 + 3.5 + 32.0) / (12.0 + 8.0 + 12.0)

        mae.reset()
        mae(predictions, new_targets)
        assert mae.get_metric() == 32.0 / 12.0
