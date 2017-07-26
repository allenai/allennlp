# pylint: disable=no-self-use,invalid-name

import numpy
from numpy.testing import assert_almost_equal
import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from allennlp.modules.similarity_functions import Bilinear
from allennlp.testing import AllenNlpTestCase

class TestBilinearSimilarityFunction(AllenNlpTestCase):
    def test_weights_are_correct_sizes(self):
        # pylint: disable=protected-access
        bilinear = Bilinear(tensor_1_dim=5, tensor_2_dim=2)
        assert list(bilinear._weight_matrix.size()) == [5, 2]
        assert list(bilinear._bias.size()) == [1]

    def test_forward_does_a_bilinear_product(self):
        # pylint: disable=protected-access
        bilinear = Bilinear(2, 2)
        bilinear._weight_matrix = Parameter(torch.FloatTensor([[-.3, .5], [2.0, -1.0]]))
        bilinear._bias = Parameter(torch.FloatTensor([.1]))
        a_vectors = Variable(torch.FloatTensor([[1, 1], [-1, -1]]))
        b_vectors = Variable(torch.FloatTensor([[1, 0], [0, 1]]))
        result = bilinear(a_vectors, b_vectors).data.numpy()
        assert result.shape == (2,)
        assert_almost_equal(result, [1.8, .6])

    def test_forward_works_with_higher_order_tensors(self):
        # pylint: disable=protected-access
        bilinear = Bilinear(4, 7)
        weights = numpy.random.rand(4, 7)
        bilinear._weight_matrix = Parameter(torch.from_numpy(weights))
        bilinear._bias = Parameter(torch.from_numpy(numpy.asarray([0])))
        a_vectors = Variable(torch.from_numpy(numpy.random.rand(5, 4, 3, 6, 4)))
        b_vectors = Variable(torch.from_numpy(numpy.random.rand(5, 4, 3, 6, 7)))
        result = bilinear(a_vectors, b_vectors).data.numpy()
        assert result.shape == (5, 4, 3, 6)
        expected_result = numpy.dot(numpy.dot(numpy.transpose(a_vectors[3, 2, 1, 3]), weights),
                                    b_vectors[3, 2, 1, 3])
        assert_almost_equal(result[3, 2, 1, 3], expected_result, decimal=5)
