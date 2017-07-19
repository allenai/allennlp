# pylint: disable=no-self-use,invalid-name
import torch
from torch.nn.init import constant
from allennlp.training.initializers import InitializerApplicator
from allennlp.training.regularizers import L1Regularizer, L2Regularizer, RegularizerApplicator
from allennlp.testing.test_case import AllenNlpTestCase


class TestRegularizers(AllenNlpTestCase):

    def test_l1_regularization(self):
        model = torch.nn.Sequential(
                torch.nn.Linear(5, 10),
                torch.nn.Linear(10, 5)
        )
        initializer = InitializerApplicator({"default": lambda tensor: constant(tensor, -1)})
        initializer(model)
        value = RegularizerApplicator({"": L1Regularizer(1.0)})(model)
        # 115 because of biases.
        assert value.data.numpy() == 115.0

    def test_l2_regularization(self):
        model = torch.nn.Sequential(
                torch.nn.Linear(5, 10),
                torch.nn.Linear(10, 5)
        )
        initializer = InitializerApplicator({"default": lambda tensor: constant(tensor, 0.5)})
        initializer(model)
        value = RegularizerApplicator({"": L2Regularizer(1.0)})(model)
        assert value.data.numpy() == 28.75

    def test_regularizer_applicator_respects_regex_matching(self):
        model = torch.nn.Sequential(
                torch.nn.Linear(5, 10),
                torch.nn.Linear(10, 5)
        )
        initializer = InitializerApplicator({"default": lambda tensor: constant(tensor, 1.)})
        initializer(model)
        value = RegularizerApplicator({"weight": L2Regularizer(0.5),
                                       "bias": L1Regularizer(1.0)})(model)
        assert value.data.numpy() == 65.0
