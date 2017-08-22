# pylint: disable=no-self-use, invalid-name
import logging

import numpy
import pytest
import pyhocon
import torch
from torch.autograd import Variable
from torch.nn.init import constant

from allennlp.nn import InitializerApplicator
from allennlp.nn.initializers import block_orthogonal
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.params import Params

class TestInitializers(AllenNlpTestCase):
    def setUp(self):
        super(TestInitializers, self).setUp()
        logging.getLogger('allennlp.nn.initializers').disabled = False

    def tearDown(self):
        super(TestInitializers, self).tearDown()
        logging.getLogger('allennlp.nn.initializers').disabled = True

    def test_all_parameters_are_initialized(self):
        model = torch.nn.Sequential(
                torch.nn.Linear(5, 10),
                torch.nn.Linear(10, 5)
        )
        initializer = InitializerApplicator(default_initializer=lambda tensor: constant(tensor, 5))
        initializer(model)
        for parameter in model.parameters():
            assert torch.equal(parameter.data, torch.ones(parameter.size()) * 5)

    def test_regex_matches_are_initialized_correctly(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.linear_1_with_funky_name = torch.nn.Linear(5, 10)
                self.linear_2 = torch.nn.Linear(10, 5)
                self.conv = torch.nn.Conv1d(5, 5, 5)

            def forward(self, inputs):  # pylint: disable=arguments-differ
                pass

        # pyhocon does funny things if there's a . in a key.  This test makes sure that we
        # handle these kinds of regexes correctly.
        json_params = """{
        "conv": {"type": "constant", "val": 5},
        "funky_na.*bi": {"type": "constant", "val": 7},
        "default": {"type": "constant", "val": 10}
        }
        """
        params = Params(pyhocon.ConfigFactory.parse_string(json_params))
        initializers = InitializerApplicator.from_params(params)
        model = Net()
        initializers(model)

        for parameter in model.conv.parameters():
            assert torch.equal(parameter.data, torch.ones(parameter.size()) * 5)

        parameter = model.linear_1_with_funky_name.bias
        assert torch.equal(parameter.data, torch.ones(parameter.size()) * 7)
        parameter = model.linear_1_with_funky_name.weight
        assert torch.equal(parameter.data, torch.ones(parameter.size()) * 10)

        for parameter in model.linear_2.parameters():
            assert torch.equal(parameter.data, torch.ones(parameter.size()) * 10)

    def test_from_params(self):

        to_exclude = ["this", "and", "that"]
        params = Params({
                "conv": "orthogonal",
                "linear": {
                        "type": "constant",
                        "val": 1
                },
                "exclude": to_exclude
        })
        initializer_applicator = InitializerApplicator.from_params(params)
        # pylint: disable=protected-access
        assert initializer_applicator._exclude == to_exclude
        initializers = initializer_applicator._initializers
        assert initializers["conv"]._init_function == torch.nn.init.orthogonal

        tensor = torch.FloatTensor([0, 0, 0, 0, 0])
        initializers["linear"](tensor)
        numpy.testing.assert_array_equal(tensor.numpy(), numpy.array([1, 1, 1, 1, 1]))

    def test_exclude_works_properly(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.linear1 = torch.nn.Linear(5, 10)
                self.linear2 = torch.nn.Linear(10, 5)
                self.linear2.weight.data.fill_(7)
                self.linear2.bias.data.fill_(7)

            def forward(self, inputs):  # pylint: disable=arguments-differ
                pass

        initializers = InitializerApplicator(default_initializer=lambda tensor: constant(tensor, 10),
                                             exclude=["linear2"])
        model = Net()
        initializers(model)

        for parameter in list(model.linear1.parameters()):
            assert torch.equal(parameter.data, torch.ones(parameter.size()) * 10)

        for parameter in list(model.linear2.parameters()):
            assert torch.equal(parameter.data, torch.ones(parameter.size()) * 7)

    def test_block_orthogonal_can_initialize(self):
        tensor = torch.autograd.Variable(torch.zeros([10, 6]))
        block_orthogonal(tensor, [5, 3])
        tensor = tensor.data.numpy()

        def test_block_is_orthogonal(block) -> None:
            matrix_product = block.T @ block
            numpy.testing.assert_array_almost_equal(matrix_product,
                                                    numpy.eye(matrix_product.shape[-1]), 6)
        test_block_is_orthogonal(tensor[:5, :3])
        test_block_is_orthogonal(tensor[:5, 3:])
        test_block_is_orthogonal(tensor[5:, 3:])
        test_block_is_orthogonal(tensor[5:, :3])

    def test_block_orthogonal_raises_on_mismatching_dimensions(self):
        tensor = torch.zeros([10, 6, 8])
        with pytest.raises(ConfigurationError):
            block_orthogonal(tensor, [7, 2, 1])
