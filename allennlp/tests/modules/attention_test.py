# pylint: disable=no-self-use,invalid-name
import numpy
from numpy.testing import assert_almost_equal
import torch
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.modules import Attention
from allennlp.common.testing import AllenNlpTestCase


class TestAttention(AllenNlpTestCase):
    def test_no_mask(self):
        attention = Attention()

        # Testing general non-batched case.
        vector = Variable(torch.FloatTensor([[0.3, 0.1, 0.5]]))
        matrix = Variable(torch.FloatTensor([[[0.6, 0.8, 0.1], [0.15, 0.5, 0.2]]]))

        result = attention(vector, matrix).data.numpy()
        assert_almost_equal(result, numpy.array([[0.52871835, 0.47128162]]))

        # Testing non-batched case where inputs are all 0s.
        vector = Variable(torch.FloatTensor([[0, 0, 0]]))
        matrix = Variable(torch.FloatTensor([[[0, 0, 0], [0, 0, 0]]]))

        result = attention(vector, matrix).data.numpy()
        assert_almost_equal(result, numpy.array([[0.5, 0.5]]))

    def test_masked(self):
        attention = Attention()
        # Testing general masked non-batched case.
        vector = Variable(torch.FloatTensor([[0.3, 0.1, 0.5]]))
        matrix = Variable(torch.FloatTensor([[[0.6, 0.8, 0.1], [0.15, 0.5, 0.2], [0.1, 0.4, 0.3]]]))
        mask = Variable(torch.FloatTensor([[1.0, 0.0, 1.0]]))
        result = attention(vector, matrix, mask).data.numpy()
        assert_almost_equal(result, numpy.array([[0.52248482, 0.0, 0.47751518]]))

    def test_batched_no_mask(self):
        attention = Attention()

        # Testing general batched case.
        vector = Variable(torch.FloatTensor([[0.3, 0.1, 0.5], [0.3, 0.1, 0.5]]))
        matrix = Variable(torch.FloatTensor([[[0.6, 0.8, 0.1], [0.15, 0.5, 0.2]],
                                             [[0.6, 0.8, 0.1], [0.15, 0.5, 0.2]]]))

        result = attention(vector, matrix).data.numpy()
        assert_almost_equal(result, numpy.array([[0.52871835, 0.47128162],
                                                 [0.52871835, 0.47128162]]))

    def test_batched_masked(self):
        attention = Attention()

        # Testing general masked non-batched case.
        vector = Variable(torch.FloatTensor([[0.3, 0.1, 0.5], [0.3, 0.1, 0.5]]))
        matrix = Variable(torch.FloatTensor([[[0.6, 0.8, 0.1], [0.15, 0.5, 0.2], [0.5, 0.3, 0.2]],
                                             [[0.6, 0.8, 0.1], [0.15, 0.5, 0.2], [0.5, 0.3, 0.2]]]))
        mask = Variable(torch.FloatTensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]))
        result = attention(vector, matrix, mask).data.numpy()
        assert_almost_equal(result, numpy.array([[0.52871835, 0.47128162, 0.0],
                                                 [0.50749944, 0.0, 0.49250056]]))

        # Test the case where a mask is all 0s and an input is all 0s.
        vector = Variable(torch.FloatTensor([[0.0, 0.0, 0.0], [0.3, 0.1, 0.5]]))
        matrix = Variable(torch.FloatTensor([[[0.6, 0.8, 0.1], [0.15, 0.5, 0.2], [0.5, 0.3, 0.2]],
                                             [[0.6, 0.8, 0.1], [0.15, 0.5, 0.2], [0.5, 0.3, 0.2]]]))
        mask = Variable(torch.FloatTensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))
        result = attention(vector, matrix, mask).data.numpy()
        assert_almost_equal(result, numpy.array([[0.5, 0.5, 0.0],
                                                 [0.0, 0.0, 0.0]]))

    def test_non_normalized_attention_works(self):
        attention = Attention(normalize=False)
        sentence_tensor = Variable(torch.FloatTensor([[[-1, 0, 4],
                                                       [1, 1, 1],
                                                       [-1, 0, 4],
                                                       [-1, 0, -1]]]))
        query_tensor = Variable(torch.FloatTensor([[.1, .8, .5]]))
        result = attention(query_tensor, sentence_tensor).data.numpy()
        assert_almost_equal(result, [[1.9, 1.4, 1.9, -.6]])

    def test_can_build_from_params(self):
        params = Params({'similarity_function': {'type': 'cosine'}, 'normalize': False})
        attention = Attention.from_params(params)
        # pylint: disable=protected-access
        assert attention._similarity_function.__class__.__name__ == 'CosineSimilarity'
        assert attention._normalize is False
