import pytest
import numpy
import torch
import torch.nn.init
from torch.nn.modules.rnn import LSTM
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.augmented_lstm import AugmentedLstm, AugmentedLSTMCell, BiAugmentedLstm
from allennlp.nn import InitializerApplicator, Initializer
from allennlp.nn.util import sort_batch_by_length


class TestAugmentedLSTM(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        tensor = torch.rand([5, 7, 10])
        tensor[0, 3:, :] = 0
        tensor[1, 4:, :] = 0
        tensor[2, 2:, :] = 0
        tensor[3, 6:, :] = 0
        sequence_lengths = torch.LongTensor([3, 4, 2, 6, 7])
        self.random_tensor = tensor
        self.sequence_lengths = sequence_lengths

    def test_variable_length_sequences_return_correctly_padded_outputs(self):
        sorted_tensor, sorted_sequence, _, _ = sort_batch_by_length(
            self.random_tensor, self.sequence_lengths
        )
        tensor = pack_padded_sequence(
            sorted_tensor, sorted_sequence.data.tolist(), batch_first=True
        )
        lstm = AugmentedLstm(10, 11)
        output, _ = lstm(tensor)
        output_sequence, _ = pad_packed_sequence(output, batch_first=True)

        numpy.testing.assert_array_equal(output_sequence.data[1, 6:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence.data[2, 4:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence.data[3, 3:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence.data[4, 2:, :].numpy(), 0.0)

    def test_variable_length_sequences_run_backward_return_correctly_padded_outputs(self):
        sorted_tensor, sorted_sequence, _, _ = sort_batch_by_length(
            self.random_tensor, self.sequence_lengths
        )
        tensor = pack_padded_sequence(
            sorted_tensor, sorted_sequence.data.tolist(), batch_first=True
        )
        lstm = AugmentedLstm(10, 11, go_forward=False)
        output, _ = lstm(tensor)
        output_sequence, _ = pad_packed_sequence(output, batch_first=True)

        numpy.testing.assert_array_equal(output_sequence.data[1, 6:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence.data[2, 4:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence.data[3, 3:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence.data[4, 2:, :].numpy(), 0.0)

    def test_augmented_lstm_computes_same_function_as_pytorch_lstm(self):
        augmented_lstm = AugmentedLstm(10, 11)
        pytorch_lstm = LSTM(10, 11, num_layers=1, batch_first=True)
        # Initialize all weights to be == 1.
        constant_init = Initializer.from_params(Params({"type": "constant", "val": 1.0}))
        initializer = InitializerApplicator([(".*", constant_init)])
        initializer(augmented_lstm)
        initializer(pytorch_lstm)

        initial_state = torch.zeros([1, 5, 11])
        initial_memory = torch.zeros([1, 5, 11])

        # Use bigger numbers to avoid floating point instability.
        sorted_tensor, sorted_sequence, _, _ = sort_batch_by_length(
            self.random_tensor * 5.0, self.sequence_lengths
        )
        lstm_input = pack_padded_sequence(
            sorted_tensor, sorted_sequence.data.tolist(), batch_first=True
        )

        augmented_output, augmented_state = augmented_lstm(
            lstm_input, (initial_state, initial_memory)
        )
        pytorch_output, pytorch_state = pytorch_lstm(lstm_input, (initial_state, initial_memory))
        pytorch_output_sequence, _ = pad_packed_sequence(pytorch_output, batch_first=True)
        augmented_output_sequence, _ = pad_packed_sequence(augmented_output, batch_first=True)

        numpy.testing.assert_array_almost_equal(
            pytorch_output_sequence.data.numpy(), augmented_output_sequence.data.numpy(), decimal=4
        )
        numpy.testing.assert_array_almost_equal(
            pytorch_state[0].data.numpy(), augmented_state[0].data.numpy(), decimal=4
        )
        numpy.testing.assert_array_almost_equal(
            pytorch_state[1].data.numpy(), augmented_state[1].data.numpy(), decimal=4
        )

    def test_augmented_lstm_works_with_highway_connections(self):
        augmented_lstm = AugmentedLstm(10, 11, use_highway=True)
        sorted_tensor, sorted_sequence, _, _ = sort_batch_by_length(
            self.random_tensor, self.sequence_lengths
        )
        lstm_input = pack_padded_sequence(
            sorted_tensor, sorted_sequence.data.tolist(), batch_first=True
        )
        augmented_lstm(lstm_input)

    def test_augmented_lstm_throws_error_on_non_packed_sequence_input(self):
        lstm = AugmentedLstm(3, 5)
        tensor = torch.rand([5, 7, 9])
        with pytest.raises(ConfigurationError):
            lstm(tensor)

    def test_augmented_lstm_is_initialized_with_correct_biases(self):
        lstm = AugmentedLSTMCell(2, 3)
        true_state_bias = numpy.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        numpy.testing.assert_array_equal(lstm.state_linearity.bias.data.numpy(), true_state_bias)

        # Non-highway case.
        lstm = AugmentedLSTMCell(2, 3, use_highway=False)
        true_state_bias = numpy.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        numpy.testing.assert_array_equal(lstm.state_linearity.bias.data.numpy(), true_state_bias)

    def test_dropout_is_not_applied_to_output_or_returned_hidden_states(self):
        sorted_tensor, sorted_sequence, _, _ = sort_batch_by_length(
            self.random_tensor, self.sequence_lengths
        )
        tensor = pack_padded_sequence(
            sorted_tensor, sorted_sequence.data.tolist(), batch_first=True
        )
        lstm = AugmentedLstm(10, 11, recurrent_dropout_probability=0.5)

        output, (hidden_state, _) = lstm(tensor)
        output_sequence, _ = pad_packed_sequence(output, batch_first=True)
        # Test returned output sequence
        num_hidden_dims_zero_across_timesteps = ((output_sequence.sum(1) == 0).sum()).item()
        # If this is not True then dropout has been applied to the output of the LSTM
        assert not num_hidden_dims_zero_across_timesteps
        # Should not have dropout applied to the last hidden state as this is not used
        # within the LSTM and makes it more consistent with the `torch.nn.LSTM` where
        # dropout is not applied to any of it's output. This would also make it more
        # consistent with the Keras LSTM implementation as well.
        hidden_state = hidden_state.squeeze()
        num_hidden_dims_zero_across_timesteps = ((hidden_state == 0).sum()).item()
        assert not num_hidden_dims_zero_across_timesteps

    def test_dropout_version_is_different_to_no_dropout(self):
        augmented_lstm = AugmentedLstm(10, 11)
        dropped_augmented_lstm = AugmentedLstm(10, 11, recurrent_dropout_probability=0.9)
        # Initialize all weights to be == 1.
        constant_init = Initializer.from_params(Params({"type": "constant", "val": 0.5}))
        initializer = InitializerApplicator([(".*", constant_init)])
        initializer(augmented_lstm)
        initializer(dropped_augmented_lstm)

        initial_state = torch.randn([1, 5, 11])
        initial_memory = torch.randn([1, 5, 11])

        # If we use too bigger number like in the PyTorch test the dropout has no affect
        sorted_tensor, sorted_sequence, _, _ = sort_batch_by_length(
            self.random_tensor, self.sequence_lengths
        )
        lstm_input = pack_padded_sequence(
            sorted_tensor, sorted_sequence.data.tolist(), batch_first=True
        )

        augmented_output, augmented_state = augmented_lstm(
            lstm_input, (initial_state, initial_memory)
        )
        dropped_output, dropped_state = dropped_augmented_lstm(
            lstm_input, (initial_state, initial_memory)
        )
        dropped_output_sequence, _ = pad_packed_sequence(dropped_output, batch_first=True)
        augmented_output_sequence, _ = pad_packed_sequence(augmented_output, batch_first=True)
        with pytest.raises(AssertionError):
            numpy.testing.assert_array_almost_equal(
                dropped_output_sequence.data.numpy(),
                augmented_output_sequence.data.numpy(),
                decimal=4,
            )
        with pytest.raises(AssertionError):
            numpy.testing.assert_array_almost_equal(
                dropped_state[0].data.numpy(), augmented_state[0].data.numpy(), decimal=4
            )
        with pytest.raises(AssertionError):
            numpy.testing.assert_array_almost_equal(
                dropped_state[1].data.numpy(), augmented_state[1].data.numpy(), decimal=4
            )

    def test_biaugmented_lstm(self):
        for bidirectional in [True, False]:
            bi_augmented_lstm = BiAugmentedLstm(
                10, 11, 3, recurrent_dropout_probability=0.1, bidirectional=bidirectional
            )
            sorted_tensor, sorted_sequence, _, _ = sort_batch_by_length(
                self.random_tensor, self.sequence_lengths
            )
            lstm_input = pack_padded_sequence(
                sorted_tensor, sorted_sequence.data.tolist(), batch_first=True
            )
            bi_augmented_lstm(lstm_input)
