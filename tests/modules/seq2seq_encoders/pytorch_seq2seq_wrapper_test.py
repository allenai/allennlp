import numpy
from numpy.testing import assert_almost_equal
import pytest
import torch
from torch.nn import LSTM, GRU
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.nn.util import sort_batch_by_length, get_lengths_from_binary_sequence_mask


class TestPytorchSeq2SeqWrapper(AllenNlpTestCase):
    def test_get_dimension_is_correct(self):
        lstm = LSTM(bidirectional=True, num_layers=3, input_size=2, hidden_size=7, batch_first=True)
        encoder = PytorchSeq2SeqWrapper(lstm)
        assert encoder.get_output_dim() == 14
        assert encoder.get_input_dim() == 2
        lstm = LSTM(
            bidirectional=False, num_layers=3, input_size=2, hidden_size=7, batch_first=True
        )
        encoder = PytorchSeq2SeqWrapper(lstm)
        assert encoder.get_output_dim() == 7
        assert encoder.get_input_dim() == 2

    def test_forward_works_even_with_empty_sequences(self):
        lstm = LSTM(bidirectional=True, num_layers=3, input_size=3, hidden_size=7, batch_first=True)
        encoder = PytorchSeq2SeqWrapper(lstm)

        tensor = torch.rand([5, 7, 3])
        tensor[1, 6:, :] = 0
        tensor[2, :, :] = 0
        tensor[3, 2:, :] = 0
        tensor[4, :, :] = 0
        mask = torch.ones(5, 7).bool()
        mask[1, 6:] = False
        mask[2, :] = False
        mask[3, 2:] = False
        mask[4, :] = False

        results = encoder(tensor, mask)

        for i in (0, 1, 3):
            assert not (results[i] == 0.0).data.all()
        for i in (2, 4):
            assert (results[i] == 0.0).data.all()

    def test_forward_pulls_out_correct_tensor_without_sequence_lengths(self):
        lstm = LSTM(bidirectional=True, num_layers=3, input_size=2, hidden_size=7, batch_first=True)
        encoder = PytorchSeq2SeqWrapper(lstm)
        input_tensor = torch.FloatTensor([[[0.7, 0.8], [0.1, 1.5]]])
        lstm_output = lstm(input_tensor)
        encoder_output = encoder(input_tensor, None)
        assert_almost_equal(encoder_output.data.numpy(), lstm_output[0].data.numpy())

    def test_forward_pulls_out_correct_tensor_with_sequence_lengths(self):
        lstm = LSTM(bidirectional=True, num_layers=3, input_size=3, hidden_size=7, batch_first=True)
        encoder = PytorchSeq2SeqWrapper(lstm)
        input_tensor = torch.rand([5, 7, 3])
        input_tensor[1, 6:, :] = 0
        input_tensor[2, 4:, :] = 0
        input_tensor[3, 2:, :] = 0
        input_tensor[4, 1:, :] = 0
        mask = torch.ones(5, 7).bool()
        mask[1, 6:] = False
        mask[2, 4:] = False
        mask[3, 2:] = False
        mask[4, 1:] = False

        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        packed_sequence = pack_padded_sequence(
            input_tensor, sequence_lengths.data.tolist(), batch_first=True
        )
        lstm_output, _ = lstm(packed_sequence)
        encoder_output = encoder(input_tensor, mask)
        lstm_tensor, _ = pad_packed_sequence(lstm_output, batch_first=True)
        assert_almost_equal(encoder_output.data.numpy(), lstm_tensor.data.numpy())

    def test_forward_pulls_out_correct_tensor_for_unsorted_batches(self):
        lstm = LSTM(bidirectional=True, num_layers=3, input_size=3, hidden_size=7, batch_first=True)
        encoder = PytorchSeq2SeqWrapper(lstm)
        input_tensor = torch.rand([5, 7, 3])
        input_tensor[0, 3:, :] = 0
        input_tensor[1, 4:, :] = 0
        input_tensor[2, 2:, :] = 0
        input_tensor[3, 6:, :] = 0
        mask = torch.ones(5, 7).bool()
        mask[0, 3:] = False
        mask[1, 4:] = False
        mask[2, 2:] = False
        mask[3, 6:] = False

        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        sorted_inputs, sorted_sequence_lengths, restoration_indices, _ = sort_batch_by_length(
            input_tensor, sequence_lengths
        )
        packed_sequence = pack_padded_sequence(
            sorted_inputs, sorted_sequence_lengths.data.tolist(), batch_first=True
        )
        lstm_output, _ = lstm(packed_sequence)
        encoder_output = encoder(input_tensor, mask)
        lstm_tensor, _ = pad_packed_sequence(lstm_output, batch_first=True)
        assert_almost_equal(
            encoder_output.data.numpy(),
            lstm_tensor.index_select(0, restoration_indices).data.numpy(),
        )

    def test_forward_does_not_compress_tensors_padded_to_greater_than_the_max_sequence_length(self):

        lstm = LSTM(bidirectional=True, num_layers=3, input_size=3, hidden_size=7, batch_first=True)
        encoder = PytorchSeq2SeqWrapper(lstm)
        input_tensor = torch.rand([5, 8, 3])
        input_tensor[:, 7, :] = 0
        mask = torch.ones(5, 8).bool()
        mask[:, 7] = False

        encoder_output = encoder(input_tensor, mask)
        assert encoder_output.size(1) == 8

    def test_wrapper_raises_if_batch_first_is_false(self):

        with pytest.raises(ConfigurationError):
            lstm = LSTM(bidirectional=True, num_layers=3, input_size=3, hidden_size=7)
            _ = PytorchSeq2SeqWrapper(lstm)

    def test_wrapper_works_when_passed_state_with_zero_length_sequences(self):
        lstm = LSTM(bidirectional=True, num_layers=3, input_size=3, hidden_size=7, batch_first=True)
        encoder = PytorchSeq2SeqWrapper(lstm)
        input_tensor = torch.rand([5, 7, 3])
        mask = torch.ones(5, 7).bool()
        mask[0, 3:] = False
        mask[1, 4:] = False
        mask[2, 0:] = False
        mask[3, 6:] = False

        # Initial states are of shape (num_layers * num_directions, batch_size, hidden_dim)
        initial_states = torch.randn(6, 5, 7), torch.randn(6, 5, 7)

        _ = encoder(input_tensor, mask, initial_states)

    def test_wrapper_can_call_backward_with_zero_length_sequences(self):
        lstm = LSTM(bidirectional=True, num_layers=3, input_size=3, hidden_size=7, batch_first=True)
        encoder = PytorchSeq2SeqWrapper(lstm)
        input_tensor = torch.rand([5, 7, 3])
        mask = torch.ones(5, 7).bool()
        mask[0, 3:] = False
        mask[1, 4:] = False
        mask[2, 0:] = 0  # zero length False
        mask[3, 6:] = False

        output = encoder(input_tensor, mask)

        output.sum().backward()

    def test_wrapper_stateful(self):
        lstm = LSTM(bidirectional=True, num_layers=2, input_size=3, hidden_size=7, batch_first=True)
        encoder = PytorchSeq2SeqWrapper(lstm, stateful=True)

        # To test the stateful functionality we need to call the encoder multiple times.
        # Different batch sizes further tests some of the logic.
        batch_sizes = [5, 10, 8]
        sequence_lengths = [4, 6, 7]
        states = []
        for batch_size, sequence_length in zip(batch_sizes, sequence_lengths):
            tensor = torch.rand([batch_size, sequence_length, 3])
            mask = torch.ones(batch_size, sequence_length).bool()
            mask.data[0, 3:] = 0
            encoder_output = encoder(tensor, mask)
            states.append(encoder._states)

        # Check that the output is masked properly.
        assert_almost_equal(encoder_output[0, 3:, :].data.numpy(), numpy.zeros((4, 14)))

        for k in range(2):
            assert_almost_equal(
                states[-1][k][:, -2:, :].data.numpy(), states[-2][k][:, -2:, :].data.numpy()
            )

    def test_wrapper_stateful_single_state_gru(self):
        gru = GRU(bidirectional=True, num_layers=2, input_size=3, hidden_size=7, batch_first=True)
        encoder = PytorchSeq2SeqWrapper(gru, stateful=True)

        batch_sizes = [10, 5]
        states = []
        for batch_size in batch_sizes:
            tensor = torch.rand([batch_size, 5, 3])
            mask = torch.ones(batch_size, 5).bool()
            mask.data[0, 3:] = 0
            encoder_output = encoder(tensor, mask)
            states.append(encoder._states)

        assert_almost_equal(encoder_output[0, 3:, :].data.numpy(), numpy.zeros((2, 14)))
        assert_almost_equal(
            states[-1][0][:, -5:, :].data.numpy(), states[-2][0][:, -5:, :].data.numpy()
        )
