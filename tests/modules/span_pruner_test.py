# pylint: disable=no-self-use,invalid-name
import torch
from torch.autograd import Variable

from allennlp.modules import SpanPruner
from allennlp.common.testing import AllenNlpTestCase


class TestSpanPruner(AllenNlpTestCase):
    def test_forward_works_on_simple_input(self):

        scorer = torch.nn.Linear(5, 1)

        pruner = SpanPruner(scorer=scorer)

        spans = Variable(torch.randn([3, 20, 5]))
        mask = Variable(torch.ones([3,20]))

        pruned_spans = pruner(spans, mask, 2)