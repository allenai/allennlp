# pylint: disable=no-self-use,invalid-name
from numpy.testing import assert_almost_equal
import torch
from torch.autograd import Variable
from torch.nn import Embedding, Parameter

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules import TimeDistributed


class TestTimeDistributed(AllenNlpTestCase):
    def test_time_distributed_reshapes_correctly(self):
        char_embedding = Embedding(2, 2)
        char_embedding.weight = Parameter(torch.FloatTensor([[.4, .4], [.5, .5]]))
        distributed_embedding = TimeDistributed(char_embedding)
        char_input = Variable(torch.LongTensor([[[1, 0], [1, 1]]]))
        output = distributed_embedding(char_input)
        assert_almost_equal(output.data.numpy(),
                            [[[[.5, .5], [.4, .4]], [[.5, .5,], [.5, .5]]]])

    def test_time_distributed_works_with_multiple_inputs(self):
        module = lambda x, y: x + y
        distributed = TimeDistributed(module)
        x_input = Variable(torch.LongTensor([[[1, 2], [3, 4]]]))
        y_input = Variable(torch.LongTensor([[[4, 2], [9, 1]]]))
        output = distributed(x_input, y_input)
        assert_almost_equal(output.data.numpy(), [[[5, 4], [12, 5]]])
