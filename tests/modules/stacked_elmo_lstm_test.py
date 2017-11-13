# pylint: disable=no-self-use,invalid-name
import numpy
import torch
from torch.nn.utils.rnn import pack_padded_sequence

from allennlp.modules.stacked_elmo_lstm import ElmoLstm
from allennlp.common.testing import AllenNlpTestCase


class TestElmoLstmCell(AllenNlpTestCase):
    def test_stacked_elmo_lstm(self):
        input_tensor = torch.autograd.Variable(torch.rand(4, 5, 3))
        input_tensor[1, 4:, :] = 0.
        input_tensor[2, 2:, :] = 0.
        input_tensor[3, 1:, :] = 0.
        input_tensor = pack_padded_sequence(input_tensor, [5, 4, 2, 1], batch_first=True)
        lstm = ElmoLstm(num_layers=2,
                        input_size=3,
                        hidden_size=5,
                        cell_size=7,
                        memory_cell_clip_value=2,
                        state_projection_clip_value=1)
        output_sequence, lstm_state = lstm(input_tensor)
        # Check all the layer outputs are masked properly.
        numpy.testing.assert_array_equal(output_sequence.data[:, 1, 4:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence.data[:, 2, 2:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence.data[:, 3, 1:, :].numpy(), 0.0)

        # LSTM state should be (num_layers, batch_size, hidden_size)
        assert list(lstm_state[0].size()) == [2, 4, 10]
        # LSTM memory cell should be (num_layers, batch_size, cell_size)
        assert list((lstm_state[1].size())) == [2, 4, 14]
