# pylint: disable=no-self-use,invalid-name
from numpy.testing import assert_allclose
import torch
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.modules import MatrixAttention
from allennlp.common.testing import AllenNlpTestCase


class TestMatrixAttention(AllenNlpTestCase):
    def test_forward_works_on_simple_input(self):
        attention = MatrixAttention()
        sentence_1_tensor = Variable(torch.FloatTensor([[[1, 1, 1], [-1, 0, 1]]]))
        sentence_2_tensor = Variable(torch.FloatTensor([[[1, 1, 1], [-1, 0, 1], [-1, -1, -1]]]))
        result = attention(sentence_1_tensor, sentence_2_tensor).data.numpy()
        assert result.shape == (1, 2, 3)
        assert_allclose(result, [[[3, 0, -3], [0, 2, 0]]])

    def test_can_build_from_params(self):
        params = Params({'similarity_function': {'type': 'cosine'}})
        attention = MatrixAttention.from_params(params)
        # pylint: disable=protected-access
        assert attention._similarity_function.__class__.__name__ == 'CosineSimilarity'
