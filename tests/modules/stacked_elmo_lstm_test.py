import numpy
import torch

from allennlp.modules.elmo_lstm import ElmoLstm
from allennlp.common.testing import AllenNlpTestCase


class TestElmoLstmCell(AllenNlpTestCase):
    def test_elmo_lstm(self):
        input_tensor = torch.rand(4, 5, 3)
        input_tensor[1, 4:, :] = 0.0
        input_tensor[2, 2:, :] = 0.0
        input_tensor[3, 1:, :] = 0.0
        mask = torch.ones([4, 5]).bool()
        mask[1, 4:] = False
        mask[2, 2:] = False
        mask[3, 1:] = False

        lstm = ElmoLstm(
            num_layers=2,
            input_size=3,
            hidden_size=5,
            cell_size=7,
            memory_cell_clip_value=2,
            state_projection_clip_value=1,
        )
        output_sequence = lstm(input_tensor, mask)

        # Check all the layer outputs are masked properly.
        numpy.testing.assert_array_equal(output_sequence.data[:, 1, 4:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence.data[:, 2, 2:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence.data[:, 3, 1:, :].numpy(), 0.0)

        # LSTM state should be (num_layers, batch_size, hidden_size)
        assert list(lstm._states[0].size()) == [2, 4, 10]
        # LSTM memory cell should be (num_layers, batch_size, cell_size)
        assert list((lstm._states[1].size())) == [2, 4, 14]
