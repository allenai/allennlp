import numpy
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from allennlp.modules.stacked_alternating_lstm import StackedAlternatingLstm
from allennlp.common.testing import AllenNlpTestCase


class TestStackedAlternatingLstm(AllenNlpTestCase):
    def test_stacked_alternating_lstm_completes_forward_pass(self):
        input_tensor = torch.rand(4, 5, 3)
        input_tensor[1, 4:, :] = 0.0
        input_tensor[2, 2:, :] = 0.0
        input_tensor[3, 1:, :] = 0.0
        input_tensor = pack_padded_sequence(input_tensor, [5, 4, 2, 1], batch_first=True)
        lstm = StackedAlternatingLstm(3, 7, 3)
        output, _ = lstm(input_tensor)
        output_sequence, _ = pad_packed_sequence(output, batch_first=True)
        numpy.testing.assert_array_equal(output_sequence.data[1, 4:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence.data[2, 2:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence.data[3, 1:, :].numpy(), 0.0)

    def test_lstms_are_interleaved(self):
        lstm = StackedAlternatingLstm(3, 7, 8)
        for i, layer in enumerate(lstm.lstm_layers):
            if i % 2 == 0:
                assert layer.go_forward
            else:
                assert not layer.go_forward
