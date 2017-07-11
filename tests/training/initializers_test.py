
import torch
import pytest
from allennlp.training.initializers import Constant, InitializerApplicator
from allennlp.testing.test_case import AllenNlpTestCase
from allennlp.common.checks import ConfigurationError


class TestInitializers(AllenNlpTestCase):

    def test_all_parameters_are_initialized(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(5, 10),
            torch.nn.Linear(10, 5)
        )
        initializer = InitializerApplicator([Constant(5)])
        initializer(model)
        for parameter in model.parameters():
            assert torch.equal(parameter.data, torch.ones(parameter.size()) * 5)

    def test_multiple_initializers_with_no_regex_raises(self):
        with pytest.raises(ConfigurationError):
            _ = InitializerApplicator([Constant(5), Constant(10)])

    def test_regex_matches_are_initialized_correctly(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(5, 10),
            torch.nn.Linear(10, 5),
            torch.nn.Conv1d(5, 5, 5)
        )
        initializers = InitializerApplicator([Constant(10), Constant(5, "Conv")])
        initializers(model)
        layers = list(model.children())
        linear_layers = layers[:2]
        conv_layer = model[2]

        for parameter in list(linear_layers[0].parameters()) + list(linear_layers[1].parameters()):
            assert torch.equal(parameter.data, torch.ones(parameter.size()) * 10)

        for parameter in conv_layer.parameters():
            assert torch.equal(parameter.data, torch.ones(parameter.size()) * 5)
