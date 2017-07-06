# pylint: disable=no-self-use,invalid-name
from numpy.testing import assert_almost_equal
import torch
from torch.autograd import Variable
from torch.nn import Embedding, Parameter

from allennlp.testing.test_case import AllenNlpTestCase
from allennlp.layers import TimeDistributed


class TestTimeDistributed(AllenNlpTestCase):
    def test_time_distributed_reshapes_correctly(self):
        char_embedding = Embedding(2, 2)
        char_embedding.weight = Parameter(torch.FloatTensor([[.4, .4], [.5, .5]]))
        distributed_embedding = TimeDistributed(char_embedding)
        char_input = Variable(torch.LongTensor([[[1, 0], [1, 1]]]))
        output = distributed_embedding(char_input)
        assert_almost_equal(output.data.numpy(),
                            [[[[.5, .5], [.4, .4]], [[.5, .5,], [.5, .5]]]])
