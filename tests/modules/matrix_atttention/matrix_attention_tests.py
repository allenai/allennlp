import torch
from numpy.testing import assert_almost_equal
import numpy as np
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.modules.matrix_attention import CosineMatrixAttention, DotProductMatrixAttention


class MatrixAttentionTests(AllenNlpTestCase):

    def test_cosine_similarity_matrix_for_2d(self):
        # example use case: a batch of size 2, each batch consists of a vector of size 3
        # and want to see how similar they are to another input matrix
        output = CosineMatrixAttention()(torch.FloatTensor([[2, 4, 6], [0, 0, 0]]),
                                         torch.FloatTensor([[1, 2, 3], [4, 5, 6]]))

        # notice that it does not take magnitude into account
        assert_almost_equal(output.numpy(), np.array([0.99, 0]), decimal=2)

    def test_cosine_similarity_matrix_for_3d(self):
        # example use case: a batch of size 2,
        # with a time element component (e.g. sentences of length 2) each word is a vector of length 3.
        # it is comparing this with another input of the same type
        output = CosineMatrixAttention()(torch.FloatTensor([[[0, 0, 0], [4, 5, 6]], [[-7, -8, -9], [10, 11, 12]]]),
                                         torch.FloatTensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))

        # for the first batch there is no correlation between the first words of the input matrix,
        # but perfect correlation for the second word
        # for the second batch there is negative correlation for the first words
        # and there is a correlation for the second word.
        assert_almost_equal(output.numpy(), np.array([[0, 1], [-1, 1]]), decimal=2)


    def test_dot_product_simialrity_for_2d(self):
        # example use case: a batch of size 2, each batch consists of a vector of size 3
        # and want to see how similar they are to another input matrix
        output = DotProductMatrixAttention()(torch.FloatTensor([[2, 4, 6], [0, 0, 0]]),
                                             torch.FloatTensor([[1, 2, 3], [4, 5, 6]]))

        # notice that it does not take magnitude into account
        assert_almost_equal(output.numpy(), np.array([0.99, 0]), decimal=2)

    def test_dot_product_similarity_matrix_for_3d(self):
        # example use case: a batch of size 2,
        # with a time element component (e.g. sentences of length 2) each word is a vector of length 3.
        # it is comparing this with another input of the same type
        output = DotProductMatrixAttention()(torch.FloatTensor([[[0, 0, 0], [4, 5, 6]]]),
                                             torch.FloatTensor([[[1, 2, 3], [4, 5, 6]]]))

        # for the first batch there is no correlation between the first words of the input matrix,
        # but perfect correlation for the second word
        # for the second batch there is negative correlation for the first words
        # and there is a correlation for the second word.
        assert_almost_equal(output.numpy(), np.array([[0, 1], [-1, 1]]), decimal=2)

