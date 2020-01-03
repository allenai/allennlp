import numpy
from numpy.testing import assert_almost_equal
import torch

from allennlp.common import Params
from allennlp.modules.similarity_functions import DotProductSimilarity
from allennlp.common.testing import AllenNlpTestCase


class TestDotProductSimilarityFunction(AllenNlpTestCase):
    def test_forward_does_a_dot_product(self):
        dot_product = DotProductSimilarity()
        a_vectors = torch.LongTensor([[1, 1, 1], [-1, -1, -1]])
        b_vectors = torch.LongTensor([[1, 0, 1], [1, 0, 0]])
        result = dot_product(a_vectors, b_vectors).data.numpy()
        assert result.shape == (2,)
        assert numpy.all(result == [2, -1])

    def test_forward_works_with_higher_order_tensors(self):
        dot_product = DotProductSimilarity()
        a_vectors = numpy.random.rand(5, 4, 3, 6, 7)
        b_vectors = numpy.random.rand(5, 4, 3, 6, 7)
        desired_result = numpy.sum(a_vectors * b_vectors, axis=-1)
        result = dot_product(torch.from_numpy(a_vectors), torch.from_numpy(b_vectors)).data.numpy()
        assert result.shape == (5, 4, 3, 6)
        # We're cutting this down here with a random partial index, so that if this test fails the
        # output isn't so huge and slow.
        assert_almost_equal(result[2, 3, 1], desired_result[2, 3, 1])

    def test_can_construct_from_params(self):
        assert (
            DotProductSimilarity.from_params(Params({})).__class__.__name__
            == "DotProductSimilarity"
        )
