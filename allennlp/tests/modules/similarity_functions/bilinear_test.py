# pylint: disable=no-self-use,invalid-name

import numpy
from numpy.testing import assert_almost_equal
import torch
from torch.nn.parameter import Parameter

from allennlp.common import Params
from allennlp.modules.similarity_functions import BilinearSimilarity
from allennlp.common.testing import AllenNlpTestCase

class TestBilinearSimilarityFunction(AllenNlpTestCase):
    def test_weights_are_correct_sizes(self):
        # pylint: disable=protected-access
        bilinear = BilinearSimilarity(tensor_1_dim=5, tensor_2_dim=2)
        assert list(bilinear._weight_matrix.size()) == [5, 2]
        assert list(bilinear._bias.size()) == [1]

    def test_forward_does_a_bilinear_product(self):
        # pylint: disable=protected-access
        bilinear = BilinearSimilarity(2, 2)
        bilinear._weight_matrix = Parameter(torch.FloatTensor([[-.3, .5], [2.0, -1.0]]))
        bilinear._bias = Parameter(torch.FloatTensor([.1]))
        a_vectors = torch.FloatTensor([[1, 1], [-1, -1]])
        b_vectors = torch.FloatTensor([[1, 0], [0, 1]])
        result = bilinear(a_vectors, b_vectors).data.numpy()
        assert result.shape == (2,)
        assert_almost_equal(result, [1.8, .6])

    def test_forward_works_with_higher_order_tensors(self):
        # pylint: disable=protected-access
        bilinear = BilinearSimilarity(4, 7)
        weights = numpy.random.rand(4, 7)
        bilinear._weight_matrix = Parameter(torch.from_numpy(weights).float())
        bilinear._bias = Parameter(torch.from_numpy(numpy.asarray([0])).float())
        a_vectors = numpy.random.rand(5, 4, 3, 6, 4)
        b_vectors = numpy.random.rand(5, 4, 3, 6, 7)
        a_variables = torch.from_numpy(a_vectors).float()
        b_variables = torch.from_numpy(b_vectors).float()
        result = bilinear(a_variables, b_variables).data.numpy()
        assert result.shape == (5, 4, 3, 6)
        expected_result = numpy.dot(numpy.dot(numpy.transpose(a_vectors[3, 2, 1, 3]), weights),
                                    b_vectors[3, 2, 1, 3])
        assert_almost_equal(result[3, 2, 1, 3], expected_result, decimal=5)

    def test_can_construct_from_params(self):
        params = Params({
                'tensor_1_dim': 3,
                'tensor_2_dim': 4
                })
        bilinear = BilinearSimilarity.from_params(params)
        assert list(bilinear._weight_matrix.size()) == [3, 4]  # pylint: disable=protected-access
