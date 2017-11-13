# pylint: disable=no-self-use,invalid-name
import numpy
import torch

from allennlp.modules.elmo_lstm_cell import ElmoLstmCell
from allennlp.common.testing import AllenNlpTestCase


class TestElmoLstmCell(AllenNlpTestCase):
    def test_elmo_lstm_cell_completes_forward_pass(self):
        input_tensor = torch.autograd.Variable(torch.rand(4, 5, 3))
        input_tensor[1, 4:, :] = 0.
        input_tensor[2, 2:, :] = 0.
        input_tensor[3, 1:, :] = 0.
        lstm = ElmoLstmCell(input_size=3,
                            hidden_size=5,
                            cell_size=7,
                            memory_cell_clip_value=2,
                            state_projection_clip_value=1)
        output_sequence, lstm_state = lstm(input_tensor, [5, 4, 2, 1])
        numpy.testing.assert_array_equal(output_sequence.data[1, 4:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence.data[2, 2:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence.data[3, 1:, :].numpy(), 0.0)

        # Test the state clipping.
        numpy.testing.assert_array_less(output_sequence.data.numpy(), 1.0)
        numpy.testing.assert_array_less(-output_sequence.data.numpy(), 1.0)

        # LSTM state should be (num_layers, batch_size, hidden_size)
        assert list(lstm_state[0].size()) == [1, 4, 5]
        # LSTM memory cell should be (num_layers, batch_size, cell_size)
        assert list((lstm_state[1].size())) == [1, 4, 7]

        # Test the cell clipping.
        numpy.testing.assert_array_less(lstm_state[0].data.numpy(), 2.0)
        numpy.testing.assert_array_less(-lstm_state[0].data.numpy(), 2.0)
