import pytest
import torch

import numpy

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules import GatedSum


class TestGatedSum(AllenNlpTestCase):
    def test_gated_sum_can_run_forward(self):
        a = torch.FloatTensor([1, 2, 3, 4, 5])
        b = -a + 0.1
        weight_value = 2
        gate_value = torch.sigmoid(torch.FloatTensor([1]))
        expected = gate_value * a + (1 - gate_value) * b

        with torch.no_grad():  # because we want to change the weight
            gated_sum = GatedSum(a.size(-1))
            gated_sum._gate.weight *= 0
            gated_sum._gate.weight += weight_value
            gated_sum._gate.bias *= 0

            out = gated_sum(a, b)
            numpy.testing.assert_almost_equal(expected.data.numpy(), out.data.numpy(), decimal=5)

        with pytest.raises(ValueError):
            GatedSum(a.size(-1))(a, b.unsqueeze(0))

        with pytest.raises(ValueError):
            GatedSum(100)(a, b)

    def test_input_output_dim(self):
        dim = 77
        gated_sum = GatedSum(dim)
        numpy.testing.assert_equal(gated_sum.get_input_dim(), dim)
        numpy.testing.assert_equal(gated_sum.get_output_dim(), dim)
