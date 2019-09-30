from numpy.testing import assert_almost_equal
import torch

from allennlp.modules import Highway
from allennlp.common.testing import AllenNlpTestCase


class TestHighway(AllenNlpTestCase):
    def test_forward_works_on_simple_input(self):
        highway = Highway(2, 2)

        highway._layers[0].weight.data.fill_(1)
        highway._layers[0].bias.data.fill_(0)
        highway._layers[1].weight.data.fill_(2)
        highway._layers[1].bias.data.fill_(-2)
        input_tensor = torch.FloatTensor([[-2, 1], [3, -2]])
        result = highway(input_tensor).data.numpy()
        assert result.shape == (2, 2)
        # This was checked by hand.
        assert_almost_equal(result, [[-0.0394, 0.0197], [1.7527, -0.5550]], decimal=4)

    def test_forward_works_on_nd_input(self):
        highway = Highway(2, 2)
        input_tensor = torch.ones(2, 2, 2)
        output = highway(input_tensor)
        assert output.size() == (2, 2, 2)
