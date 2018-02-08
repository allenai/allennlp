# pylint: disable=no-self-use,invalid-name
import torch
from torch.autograd import Variable

from allennlp.modules import TriLinearAttention
from allennlp.common.testing import AllenNlpTestCase


class TestTriLinearAttention(AllenNlpTestCase):
    def test_forward_works_on_simple_input(self):
        attention = TriLinearAttention(input_dim=4)

        # (1, 2, 4)
        sentence_1_tensor = Variable(torch.FloatTensor([[[1, 1, 1, 1], [-1, 0, 1, 1]]]))
        # (1, 3, 4)
        sentence_2_tensor = Variable(torch.FloatTensor([[[1, 1, 1, 1], [-1, 0, 1, 1], [-1, -1, -1, 1]]]))
        result = attention(sentence_1_tensor, sentence_2_tensor).data.numpy()
        assert result.shape == (1, 2, 3)
