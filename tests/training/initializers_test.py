# pylint: disable=no-self-use, invalid-name
import torch
from torch.nn.init import constant

from allennlp.training.initializers import InitializerApplicator
from allennlp.testing.test_case import AllenNlpTestCase


class TestInitializers(AllenNlpTestCase):

    def test_all_parameters_are_initialized(self):
        model = torch.nn.Sequential(
                torch.nn.Linear(5, 10),
                torch.nn.Linear(10, 5)
        )
        initializer = InitializerApplicator({"default": lambda tensor: constant(tensor, 5)})
        initializer(model)
        for parameter in model.parameters():
            assert torch.equal(parameter.data, torch.ones(parameter.size()) * 5)

    def test_regex_matches_are_initialized_correctly(self):

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.linear1 = torch.nn.Linear(5, 10)
                self.linear2 = torch.nn.Linear(10, 5)
                self.conv = torch.nn.Conv1d(5, 5, 5)

            def forward(self, inputs):  # pylint: disable=arguments-differ
                pass

        initializers = InitializerApplicator({"conv": lambda tensor: constant(tensor, 5),
                                              "default": lambda tensor: constant(tensor, 10)})
        model = Net()
        initializers(model)
        layers = list(model.children())
        linear_layers = layers[:2]
        conv_layer = layers[2]

        for parameter in list(linear_layers[0].parameters()) + list(linear_layers[1].parameters()):
            assert torch.equal(parameter.data, torch.ones(parameter.size()) * 10)

        for parameter in conv_layer.parameters():
            assert torch.equal(parameter.data, torch.ones(parameter.size()) * 5)
