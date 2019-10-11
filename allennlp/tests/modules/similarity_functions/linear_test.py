import numpy
from numpy.testing import assert_almost_equal
import torch
from torch.nn.parameter import Parameter

from allennlp.common import Params
from allennlp.modules.similarity_functions import LinearSimilarity
from allennlp.common.testing import AllenNlpTestCase


class TestLinearSimilarityFunction(AllenNlpTestCase):
    def test_weights_are_correct_sizes(self):
        linear = LinearSimilarity(tensor_1_dim=3, tensor_2_dim=6, combination="x,y")
        assert list(linear._weight_vector.size()) == [9]
        assert list(linear._bias.size()) == [1]

    def test_forward_does_a_weighted_product(self):
        linear = LinearSimilarity(3, 1, combination="x,y")
        linear._weight_vector = Parameter(torch.FloatTensor([-0.3, 0.5, 2.0, -1.0]))
        linear._bias = Parameter(torch.FloatTensor([0.1]))
        a_vectors = torch.FloatTensor([[[1, 1, 1], [-1, -1, 0]]])
        b_vectors = torch.FloatTensor([[[0], [1]]])
        result = linear(a_vectors, b_vectors).data.numpy()
        assert result.shape == (1, 2)
        assert_almost_equal(result, [[2.3, -1.1]])

    def test_forward_works_with_higher_order_tensors(self):
        linear = LinearSimilarity(7, 7, combination="x,y")
        weights = numpy.random.rand(14)
        linear._weight_vector = Parameter(torch.from_numpy(weights).float())
        linear._bias = Parameter(torch.FloatTensor([0.0]))
        a_vectors = numpy.random.rand(5, 4, 3, 6, 7)
        b_vectors = numpy.random.rand(5, 4, 3, 6, 7)
        result = linear(torch.from_numpy(a_vectors).float(), torch.from_numpy(b_vectors).float())
        result = result.data.numpy()
        assert result.shape == (5, 4, 3, 6)
        combined_vectors = numpy.concatenate([a_vectors[3, 2, 1, 3, :], b_vectors[3, 2, 1, 3, :]])
        expected_result = numpy.dot(combined_vectors, weights)
        assert_almost_equal(result[3, 2, 1, 3], expected_result, decimal=6)

    def test_forward_works_with_multiply_combinations(self):
        linear = LinearSimilarity(2, 2, combination="x*y")
        linear._weight_vector = Parameter(torch.FloatTensor([-0.3, 0.5]))
        linear._bias = Parameter(torch.FloatTensor([0]))
        a_vectors = torch.FloatTensor([[1, 1], [-1, -1]])
        b_vectors = torch.FloatTensor([[1, 0], [0, 1]])
        result = linear(a_vectors, b_vectors).data.numpy()
        assert result.shape == (2,)
        assert_almost_equal(result, [-0.3, -0.5])

    def test_forward_works_with_divide_combinations(self):
        linear = LinearSimilarity(2, 2, combination="x/y")
        linear._weight_vector = Parameter(torch.FloatTensor([-0.3, 0.5]))
        linear._bias = Parameter(torch.FloatTensor([0]))
        a_vectors = torch.FloatTensor([[1, 1], [-1, -1]])
        b_vectors = torch.FloatTensor([[1, 2], [2, 1]])
        result = linear(a_vectors, b_vectors).data.numpy()
        assert result.shape == (2,)
        assert_almost_equal(result, [-0.05, -0.35])

    def test_forward_works_with_add_combinations(self):
        linear = LinearSimilarity(2, 2, combination="x+y")
        linear._weight_vector = Parameter(torch.FloatTensor([-0.3, 0.5]))
        linear._bias = Parameter(torch.FloatTensor([0]))
        a_vectors = torch.FloatTensor([[1, 1], [-1, -1]])
        b_vectors = torch.FloatTensor([[1, 0], [0, 1]])
        result = linear(a_vectors, b_vectors).data.numpy()
        assert result.shape == (2,)
        assert_almost_equal(result, [-0.1, 0.3])

    def test_forward_works_with_subtract_combinations(self):
        linear = LinearSimilarity(2, 2, combination="x-y")
        linear._weight_vector = Parameter(torch.FloatTensor([-0.3, 0.5]))
        linear._bias = Parameter(torch.FloatTensor([0]))
        a_vectors = torch.FloatTensor([[1, 1], [-1, -1]])
        b_vectors = torch.FloatTensor([[1, 0], [0, 1]])
        result = linear(a_vectors, b_vectors).data.numpy()
        assert result.shape == (2,)
        assert_almost_equal(result, [0.5, -0.7])

    def test_can_construct_from_params(self):
        params = Params({"tensor_1_dim": 4, "tensor_2_dim": 4, "combination": "x,y,x*y,y-x"})
        linear = LinearSimilarity.from_params(params)
        assert list(linear._weight_vector.size()) == [16]
