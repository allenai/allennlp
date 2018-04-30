# pylint: disable=no-self-use,invalid-name
import numpy
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from allennlp.modules.stacked_bidirectional_lstm import StackedBidirectionalLstm
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.params import Params


class TestStackedBidirectionalLstm(AllenNlpTestCase):
    def test_stacked_bidirectional_lstm_completes_forward_pass(self):
        input_tensor = torch.autograd.Variable(torch.rand(4, 5, 3))
        input_tensor[1, 4:, :] = 0.
        input_tensor[2, 2:, :] = 0.
        input_tensor[3, 1:, :] = 0.
        input_tensor = pack_padded_sequence(input_tensor, [5, 4, 2, 1], batch_first=True)
        lstm = StackedBidirectionalLstm(3, 7, 3)
        output, _ = lstm(input_tensor)
        output_sequence, _ = pad_packed_sequence(output, batch_first=True)
        numpy.testing.assert_array_equal(output_sequence.data[1, 4:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence.data[2, 2:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence.data[3, 1:, :].numpy(), 0.0)

    def test_stacked_bidirectional_lstm_can_build_from_params(self):
        params = Params({"type": "stacked_bidirectional_lstm",
                         "input_size": 5,
                         "hidden_size": 9,
                         "num_layers": 3})
        encoder = Seq2SeqEncoder.from_params(params)

        assert encoder.get_input_dim() == 5
        assert encoder.get_output_dim() == 18
        assert encoder.is_bidirectional
