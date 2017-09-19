# pylint: disable=no-self-use,invalid-name
from numpy.testing import assert_almost_equal
import pytest
import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.similarity_functions import MultiHeadedSimilarity

class TestMultiHeadedSimilarityFunction(AllenNlpTestCase):
    def test_weights_are_correct_sizes(self):
        # pylint: disable=protected-access
        similarity = MultiHeadedSimilarity(num_heads=3, tensor_1_dim=9, tensor_1_projected_dim=6,
                                           tensor_2_dim=6, tensor_2_projected_dim=12)
        assert list(similarity._tensor_1_projection.size()) == [9, 6]
        assert list(similarity._tensor_2_projection.size()) == [6, 12]
        with pytest.raises(ConfigurationError):
            similarity = MultiHeadedSimilarity(num_heads=3, tensor_1_dim=10)
        with pytest.raises(ConfigurationError):
            params = Params({'num_heads': 3, 'tensor_1_dim': 9, 'tensor_2_dim': 10})
            MultiHeadedSimilarity.from_params(params)

    def test_forward(self):
        # pylint: disable=protected-access
        similarity = MultiHeadedSimilarity(num_heads=3, tensor_1_dim=6)
        similarity._tensor_1_projection = Parameter(torch.eye(6))
        similarity._tensor_2_projection = Parameter(torch.eye(6))
        a_vectors = Variable(torch.FloatTensor([[[[1, 1, -1, -1, 0, 1], [-2, 5, 9, -1, 3, 4]]]]))
        b_vectors = Variable(torch.FloatTensor([[[[1, 1, 1, 0, 2, 5], [0, 1, -1, -7, 1, 2]]]]))
        result = similarity(a_vectors, b_vectors).data.numpy()
        assert result.shape == (1, 1, 2, 3)
        assert_almost_equal(result, [[[[2, -1, 5], [5, -2, 11]]]])
