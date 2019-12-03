import torch
from numpy.testing import assert_almost_equal
import numpy

from allennlp.common import Params
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.modules.matrix_attention import CosineMatrixAttention
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention


class TestCosineMatrixAttention(AllenNlpTestCase):
    def test_can_init_cosine(self):
        legacy_attention = MatrixAttention.from_params(Params({"type": "cosine"}))
        isinstance(legacy_attention, CosineMatrixAttention)

    def test_cosine_similarity(self):
        # example use case: a batch of size 2.
        # With a time element component (e.g. sentences of length 2) each word is a vector of length 3.
        # It is comparing this with another input of the same type
        output = CosineMatrixAttention()(
            torch.FloatTensor([[[0, 0, 0], [4, 5, 6]], [[-7, -8, -9], [10, 11, 12]]]),
            torch.FloatTensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
        )

        # For the first batch there is
        #       no correlation between the first words of the input matrix
        #       but perfect correlation for the second word
        # For the second batch there is
        #     negative correlation for the first words
        #     correlation for the second word
        assert_almost_equal(
            output.numpy(), numpy.array([[[0, 0], [0.97, 1]], [[-1, -0.99], [0.99, 1]]]), decimal=2
        )
