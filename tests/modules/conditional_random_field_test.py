# pylint: disable=no-self-use,invalid-name
from numpy.testing import assert_almost_equal
import pytest
import torch
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.modules import ConditionalRandomField
from allennlp.nn import InitializerApplicator
from allennlp.common.testing import AllenNlpTestCase


class TestConditionalRandomField(AllenNlpTestCase):
    def test_forward_works_without_mask(self):
        inputs = Variable(torch.Tensor([
                [[0, 0, .5, .5, .2], [0, 0, .3, .3, .1], [0, 0, .9, 10, 1]],
                [[0, 0, .2, .5, .2], [0, 0, 3, .3, .1], [0, 0, .9, 1, 1]],
        ]))
        labels = Variable(torch.LongTensor([
                [2, 3, 4],
                [3, 2, 2]
        ]))

        ConditionalRandomField(5, 0, 1).forward(inputs, labels)

    def test_forward_works_with_mask(self):
        inputs = Variable(torch.Tensor([
                [[0, 0, .5, .5, .2], [0, 0, .3, .3, .1], [0, 0, .9, 10, 1]],
                [[0, 0, .2, .5, .2], [0, 0, 3, .3, .1], [0, 0, .9, 1, 1]],
        ]))
        labels = Variable(torch.LongTensor([
                [2, 3, 4],
                [3, 2, 2]
        ]))
        mask = Variable(torch.ByteTensor([
                [1, 1, 1],
                [1, 1, 0]
        ]))

        ConditionalRandomField(5, 0, 1).forward(inputs, labels, mask)
