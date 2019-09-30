import numpy
from numpy.testing import assert_almost_equal
import torch

from allennlp.common import Params
from allennlp.modules.similarity_functions import CosineSimilarity
from allennlp.common.testing import AllenNlpTestCase


class TestCosineSimilarityFunction(AllenNlpTestCase):
    def test_forward_does_a_cosine_similarity(self):
        cosine_similarity = CosineSimilarity()
        a_vectors = numpy.random.rand(1, 2, 3)
        b_vectors = numpy.random.rand(1, 2, 3)
        normed_a = a_vectors / numpy.expand_dims(numpy.linalg.norm(a_vectors, 2, -1), -1)
        normed_b = b_vectors / numpy.expand_dims(numpy.linalg.norm(b_vectors, 2, -1), -1)
        desired_result = numpy.sum(normed_a * normed_b, axis=-1)
        result = cosine_similarity(
            torch.from_numpy(a_vectors), torch.from_numpy(b_vectors)
        ).data.numpy()
        assert result.shape == (1, 2)
        assert desired_result.shape == (1, 2)
        assert_almost_equal(result, desired_result)

    def test_forward_works_with_higher_order_tensors(self):
        a_vectors = numpy.random.rand(5, 4, 3, 6, 7)
        b_vectors = numpy.random.rand(5, 4, 3, 6, 7)
        cosine_similarity = CosineSimilarity()
        normed_a = a_vectors / numpy.expand_dims(numpy.linalg.norm(a_vectors, 2, -1), -1)
        normed_b = b_vectors / numpy.expand_dims(numpy.linalg.norm(b_vectors, 2, -1), -1)
        desired_result = numpy.sum(normed_a * normed_b, axis=-1)
        result = cosine_similarity(
            torch.from_numpy(a_vectors), torch.from_numpy(b_vectors)
        ).data.numpy()
        assert result.shape == (5, 4, 3, 6)
        # We're cutting this down here with a random partial index, so that if this test fails the
        # output isn't so huge and slow.
        assert_almost_equal(result[2, 3, 1], desired_result[2, 3, 1])

    def test_can_construct_from_params(self):
        assert CosineSimilarity.from_params(Params({})).__class__.__name__ == "CosineSimilarity"
