# pylint: disable=no-self-use,invalid-name
from numpy.testing import assert_almost_equal
import pytest
import torch
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.modules import FeedForward
from allennlp.nn import InitializerApplicator
from allennlp.common.testing import AllenNlpTestCase


class TestFeedForward(AllenNlpTestCase):
    def test_init_checks_hidden_dim_consistency(self):
        with pytest.raises(ConfigurationError):
            FeedForward(2, 4, [5, 5], 'relu')

    def test_init_checks_activation_consistency(self):
        with pytest.raises(ConfigurationError):
            FeedForward(2, 4, 5, ['relu', 'relu'])

    def test_forward_gives_correct_output(self):
        params = Params({
                'input_dim': 2,
                'hidden_dims': 3,
                'activations': 'relu',
                'num_layers': 2
                })
        feedforward = FeedForward.from_params(params)

        constant_init = lambda tensor: torch.nn.init.constant(tensor, 1.)
        initializer = InitializerApplicator([(".*", constant_init)])
        initializer(feedforward)

        input_tensor = Variable(torch.FloatTensor([[-3, 1]]))
        output = feedforward(input_tensor).data.numpy()
        assert output.shape == (1, 3)
        # This output was checked by hand - ReLU makes output after first hidden layer [0, 0, 0],
        # which then gets a bias added in the second layer to be [1, 1, 1].
        assert_almost_equal(output, [[1, 1, 1]])
