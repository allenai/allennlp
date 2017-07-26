# pylint: disable=no-self-use,invalid-name
import numpy
from numpy.testing import assert_allclose
import torch
from torch.autograd import Variable

from allennlp.modules import MatrixAttention
from allennlp.modules.similarity_functions import LinearSimilarity
from allennlp.testing import AllenNlpTestCase


class TestMatrixAttention(AllenNlpTestCase):
    def test_forward_works_on_simple_input(self):
        attention = MatrixAttention()
        sentence_1_tensor = Variable(torch.FloatTensor([[[1, 1, 1], [-1, 0, 1]]]))
        sentence_2_tensor = Variable(torch.FloatTensor([[[1, 1, 1], [-1, 0, 1], [-1, -1, -1]]]))
        result = attention(sentence_1_tensor, sentence_2_tensor).data.numpy()
        assert result.shape == (1, 2, 3)
        assert_allclose(result, [[[3, 0, -3], [0, 2, 0]]])
