# pylint: disable=no-self-use,invalid-name
import pytest
from numpy.testing import assert_almost_equal
import torch
from torch.autograd import Variable
from torch.nn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.nn.util import sort_batch_by_length, get_lengths_from_binary_sequence_mask


class TestPytorchSeq2VecWrapper(AllenNlpTestCase):
    def test_get_dimensions_is_correct(self):
        lstm = LSTM(bidirectional=True, num_layers=3, input_size=2, hidden_size=7, batch_first=True)
        encoder = PytorchSeq2VecWrapper(lstm)
        assert encoder.get_output_dim() == 14
        assert encoder.get_input_dim() == 2
        lstm = LSTM(bidirectional=False, num_layers=3, input_size=2, hidden_size=7, batch_first=True)
        encoder = PytorchSeq2VecWrapper(lstm)
        assert encoder.get_output_dim() == 7
        assert encoder.get_input_dim() == 2

    def test_forward_pulls_out_correct_tensor_without_sequence_lengths(self):
        lstm = LSTM(bidirectional=True, num_layers=3, input_size=2, hidden_size=7, batch_first=True)
        encoder = PytorchSeq2VecWrapper(lstm)
        input_tensor = Variable(torch.FloatTensor([[[.7, .8], [.1, 1.5]]]))
        lstm_output = lstm(input_tensor)
        encoder_output = encoder(input_tensor, None)
        assert_almost_equal(encoder_output.data.numpy(), lstm_output[0].data.numpy()[:, -1, :])

    def test_forward_pulls_out_correct_tensor_with_sequence_lengths(self):
        lstm = LSTM(bidirectional=True, num_layers=3, input_size=3, hidden_size=7, batch_first=True)
        encoder = PytorchSeq2VecWrapper(lstm)

        tensor = torch.rand([5, 7, 3])
        tensor[1, 6:, :] = 0
        tensor[2, 4:, :] = 0
        tensor[3, 2:, :] = 0
        tensor[4, 1:, :] = 0
        mask = torch.ones(5, 7)
        mask[1, 6:] = 0
        mask[2, 4:] = 0
        mask[3, 2:] = 0
        mask[4, 1:] = 0

        input_tensor = Variable(tensor)
        mask = Variable(mask)
        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        packed_sequence = pack_padded_sequence(input_tensor, list(sequence_lengths.data), batch_first=True)
        _, state = lstm(packed_sequence)
        # Transpose output state, extract the last forward and backward states and
        # reshape to be of dimension (batch_size, 2 * hidden_size).
        reshaped_state = state[0].transpose(0, 1)[:, -2:, :].contiguous()
        explicitly_concatenated_state = torch.cat([reshaped_state[:, 0, :].squeeze(1),
                                                   reshaped_state[:, 1, :].squeeze(1)], -1)
        encoder_output = encoder(input_tensor, mask)
        assert_almost_equal(encoder_output.data.numpy(), explicitly_concatenated_state.data.numpy())

    def test_forward_works_even_with_empty_sequences(self):
        lstm = LSTM(bidirectional=True, num_layers=3, input_size=3, hidden_size=11, batch_first=True)
        encoder = PytorchSeq2VecWrapper(lstm)

        tensor = torch.autograd.Variable(torch.rand([5, 7, 3]))
        tensor[1, 6:, :] = 0
        tensor[2, :, :] = 0
        tensor[3, 2:, :] = 0
        tensor[4, :, :] = 0
        mask = torch.autograd.Variable(torch.ones(5, 7))
        mask[1, 6:] = 0
        mask[2, :] = 0
        mask[3, 2:] = 0
        mask[4, :] = 0

        results = encoder(tensor, mask)

        for i in (0, 1, 3):
            assert not (results[i] == 0.).data.all()
        for i in (2, 4):
            assert (results[i] == 0.).data.all()

    def test_forward_pulls_out_correct_tensor_with_unsorted_batches(self):
        lstm = LSTM(bidirectional=True, num_layers=3, input_size=3, hidden_size=7, batch_first=True)
        encoder = PytorchSeq2VecWrapper(lstm)

        tensor = torch.rand([5, 7, 3])
        tensor[0, 3:, :] = 0
        tensor[1, 4:, :] = 0
        tensor[2, 2:, :] = 0
        tensor[3, 6:, :] = 0
        mask = torch.ones(5, 7)
        mask[0, 3:] = 0
        mask[1, 4:] = 0
        mask[2, 2:] = 0
        mask[3, 6:] = 0

        input_tensor = Variable(tensor)
        mask = Variable(mask)
        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        sorted_inputs, sorted_sequence_lengths, restoration_indices, _ = sort_batch_by_length(input_tensor,
                                                                                              sequence_lengths)
        packed_sequence = pack_padded_sequence(sorted_inputs,
                                               sorted_sequence_lengths.data.tolist(),
                                               batch_first=True)
        _, state = lstm(packed_sequence)
        # Transpose output state, extract the last forward and backward states and
        # reshape to be of dimension (batch_size, 2 * hidden_size).
        sorted_transposed_state = state[0].transpose(0, 1).index_select(0, restoration_indices)
        reshaped_state = sorted_transposed_state[:, -2:, :].contiguous()
        explicitly_concatenated_state = torch.cat([reshaped_state[:, 0, :].squeeze(1),
                                                   reshaped_state[:, 1, :].squeeze(1)], -1)
        encoder_output = encoder(input_tensor, mask)
        assert_almost_equal(encoder_output.data.numpy(), explicitly_concatenated_state.data.numpy())

    def test_wrapper_raises_if_batch_first_is_false(self):
        with pytest.raises(ConfigurationError):
            lstm = LSTM(bidirectional=True, num_layers=3, input_size=3, hidden_size=7)
            _ = PytorchSeq2VecWrapper(lstm)
