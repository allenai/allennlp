# pylint: disable=no-self-use,invalid-name
from numpy.testing import assert_almost_equal
import torch
from torch.autograd import Variable
from torch.nn import LSTM

from allennlp.common import Params
from allennlp.modules.seq2vec_encoders import WrappedPytorchRnn
from allennlp.testing.test_case import AllenNlpTestCase


class TestWrappedPytorchSeq2VecRnn(AllenNlpTestCase):
    def test_get_output_dim_is_correct(self):
        lstm = LSTM(bidirectional=True, num_layers=3, input_size=2, hidden_size=7)
        encoder = WrappedPytorchRnn(lstm)
        assert encoder.get_output_dim() == 14
        lstm = LSTM(bidirectional=False, num_layers=3, input_size=2, hidden_size=7)
        encoder = WrappedPytorchRnn(lstm)
        assert encoder.get_output_dim() == 7

    def test_forward_pulls_out_correct_tensor(self):
        lstm = LSTM(bidirectional=True, num_layers=3, input_size=2, hidden_size=7)
        encoder = WrappedPytorchRnn(lstm)
        input_tensor = Variable(torch.FloatTensor([[[.7, .8], [.1, 1.5]]]))
        lstm_output = lstm(input_tensor)
        encoder_output = encoder(input_tensor)
        assert_almost_equal(encoder_output.data.numpy(), lstm_output[0].data.numpy()[-1])
