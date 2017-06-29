

import torch

from allennlp.training.regularizers import L1Regularizer, L2Regularizer
from allennlp.training.initializers import Constant
from allennlp.testing.test_case import AllenNlpTestCase


class TestRegularizers(AllenNlpTestCase):

    def test_l1_regularization(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(5, 10),
            torch.nn.Linear(10, 5)
        )
        initializer = Constant(-1)
        initializer(model)
        value = L1Regularizer(1.0)(model)
        print(value)
        # 115 because of biases.
        assert value.data.numpy() == 115.0

    def test_l2_regularization(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(5, 10),
            torch.nn.Linear(10, 5)
        )
        initializer = Constant(0.5)
        initializer(model)
        value = L2Regularizer(1.0)(model)
        assert value.data.numpy() == 28.75

