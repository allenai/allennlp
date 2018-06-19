# pylint: disable=no-self-use,invalid-name,protected-access
import torch
import numpy

from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.metrics import Entropy


class EntropyTest(AllenNlpTestCase):
    def test_low_entropy_distribution(self):
        metric = Entropy()
        logits = torch.Tensor([[10000, -10000, -10000, -1000],
                               [10000, -10000, -10000, -1000]])
        metric(logits)
        assert metric.get_metric() == 0.0

    def test_entropy_for_uniform_distribution(self):
        metric = Entropy()
        logits = torch.Tensor([[1, 1, 1, 1],
                               [1, 1, 1, 1]])
        metric(logits)
        numpy.testing.assert_almost_equal(metric.get_metric(), 1.38629436)
        # actual values shouldn't effect uniform distribution:
        logits = torch.Tensor([[2, 2, 2, 2],
                               [2, 2, 2, 2]])
        metric(logits)
        numpy.testing.assert_almost_equal(metric.get_metric(), 1.38629436)

        metric.reset()
        assert metric._entropy == 0.0
        assert metric._count == 0.0

    def test_masked_case(self):
        metric = Entropy()
        # This would have non-zero entropy without the mask.
        logits = torch.Tensor([[1, 1, 1, 1],
                               [10000, -10000, -10000, -1000]])
        mask = torch.Tensor([0, 1])
        metric(logits, mask)
        assert metric.get_metric() == 0.0
