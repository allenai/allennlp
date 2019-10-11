import torch
from numpy.testing import assert_almost_equal
import numpy

from allennlp.common import Params
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.modules.matrix_attention import DotProductMatrixAttention
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention


class TestDotProductMatrixAttention(AllenNlpTestCase):
    def test_can_init_dot(self):
        legacy_attention = MatrixAttention.from_params(Params({"type": "dot_product"}))
        isinstance(legacy_attention, DotProductMatrixAttention)

    def test_dot_product_similarity(self):
        # example use case: a batch of size 2,
        # with a time element component (e.g. sentences of length 2) each word is a vector of length 3.
        # it is comparing this with another input of the same type
        output = DotProductMatrixAttention()(
            torch.FloatTensor([[[0, 0, 0], [4, 5, 6]], [[-7, -8, -9], [10, 11, 12]]]),
            torch.FloatTensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
        )

        # for the first batch there is
        #       no correlation between the first words of the input matrix
        #       but perfect correlation for the second word
        # for the second batch there is
        #       negative correlation for the first words
        #       a correlation for the second word
        assert_almost_equal(
            output.numpy(), numpy.array([[[0, 0], [32, 77]], [[-194, -266], [266, 365]]]), decimal=2
        )
