import json
import logging
import math

import numpy
import pytest
import torch
import _jsonnet

from allennlp.nn import InitializerApplicator, Initializer
from allennlp.nn.initializers import block_orthogonal, uniform_unit_scaling
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.params import Params


class TestInitializers(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        logging.getLogger("allennlp.nn.initializers").disabled = False

    def tearDown(self):
        super().tearDown()
        logging.getLogger("allennlp.nn.initializers").disabled = True

    def test_from_params_string(self):
        Initializer.from_params(params="eye")

    def test_from_params_none(self):
        Initializer.from_params(params=None)

    def test_regex_matches_are_initialized_correctly(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_1_with_funky_name = torch.nn.Linear(5, 10)
                self.linear_2 = torch.nn.Linear(10, 5)
                self.conv = torch.nn.Conv1d(5, 5, 5)

            def forward(self, inputs):
                pass

        # Make sure we handle regexes properly
        json_params = """{"initializer": {"regexes": [
        ["conv", {"type": "constant", "val": 5}],
        ["funky_na.*bi", {"type": "constant", "val": 7}]
        ]}}
        """
        params = Params(json.loads(_jsonnet.evaluate_snippet("", json_params)))
        initializers = InitializerApplicator.from_params(params=params["initializer"])
        model = Net()
        initializers(model)

        for parameter in model.conv.parameters():
            assert torch.equal(parameter.data, torch.ones(parameter.size()) * 5)

        parameter = model.linear_1_with_funky_name.bias
        assert torch.equal(parameter.data, torch.ones(parameter.size()) * 7)

    def test_block_orthogonal_can_initialize(self):
        tensor = torch.zeros([10, 6])
        block_orthogonal(tensor, [5, 3])
        tensor = tensor.data.numpy()

        def test_block_is_orthogonal(block) -> None:
            matrix_product = block.T @ block
            numpy.testing.assert_array_almost_equal(
                matrix_product, numpy.eye(matrix_product.shape[-1]), 6
            )

        test_block_is_orthogonal(tensor[:5, :3])
        test_block_is_orthogonal(tensor[:5, 3:])
        test_block_is_orthogonal(tensor[5:, 3:])
        test_block_is_orthogonal(tensor[5:, :3])

    def test_block_orthogonal_raises_on_mismatching_dimensions(self):
        tensor = torch.zeros([10, 6, 8])
        with pytest.raises(ConfigurationError):
            block_orthogonal(tensor, [7, 2, 1])

    def test_uniform_unit_scaling_can_initialize(self):
        tensor = torch.zeros([10, 6])
        uniform_unit_scaling(tensor, "linear")

        assert tensor.data.max() < math.sqrt(3 / 10)
        assert tensor.data.min() > -math.sqrt(3 / 10)

        # Check that it gets the scaling correct for relu (1.43).
        uniform_unit_scaling(tensor, "relu")
        assert tensor.data.max() < math.sqrt(3 / 10) * 1.43
        assert tensor.data.min() > -math.sqrt(3 / 10) * 1.43

    def test_regex_match_prevention_prevents_and_overrides(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_1 = torch.nn.Linear(5, 10)
                self.linear_2 = torch.nn.Linear(10, 5)
                # typical actual usage: modules loaded from allenlp.model.load(..)
                self.linear_3_transfer = torch.nn.Linear(5, 10)
                self.linear_4_transfer = torch.nn.Linear(10, 5)
                self.pretrained_conv = torch.nn.Conv1d(5, 5, 5)

            def forward(self, inputs):
                pass

        json_params = """{"initializer": {
        "regexes": [
            [".*linear.*", {"type": "constant", "val": 10}],
            [".*conv.*", {"type": "constant", "val": 10}]
            ],
        "prevent_regexes": [".*_transfer.*", ".*pretrained.*"]
        }}
        """
        params = Params(json.loads(_jsonnet.evaluate_snippet("", json_params)))
        initializers = InitializerApplicator.from_params(params=params["initializer"])
        model = Net()
        initializers(model)

        for module in [model.linear_1, model.linear_2]:
            for parameter in module.parameters():
                assert torch.equal(parameter.data, torch.ones(parameter.size()) * 10)

        transfered_modules = [
            model.linear_3_transfer,
            model.linear_4_transfer,
            model.pretrained_conv,
        ]

        for module in transfered_modules:
            for parameter in module.parameters():
                assert not torch.equal(parameter.data, torch.ones(parameter.size()) * 10)
