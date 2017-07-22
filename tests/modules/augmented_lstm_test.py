# pylint: disable=invalid-name,no-self-use
import pytest
import numpy
import torch
import torch.nn.init
from torch.nn.modules.rnn import LSTM
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from allennlp.modules.augmented_lstm import AugmentedLstm
from allennlp.testing.test_case import AllenNlpTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.common.tensor import sort_batch_by_length
from allennlp.training.initializers import InitializerApplicator


class TestAugmentedLSTM(AllenNlpTestCase):
    def setUp(self):
        super(TestAugmentedLSTM, self).setUp()
        tensor = torch.rand([5, 7, 10])
        tensor[0, 3:, :] = 0
        tensor[1, 4:, :] = 0
        tensor[2, 2:, :] = 0
        tensor[3, 6:, :] = 0
        tensor = torch.autograd.Variable(tensor)
        sequence_lengths = torch.LongTensor([3, 4, 2, 6, 7])
        self.random_tensor = tensor
        self.sequence_lengths = sequence_lengths

    def test_variable_length_sequences_return_correctly_padded_outputs(self):
        sorted_tensor, sorted_sequence, _ = sort_batch_by_length(self.random_tensor, self.sequence_lengths)
        tensor = pack_padded_sequence(sorted_tensor, sorted_sequence.tolist(), batch_first=True)
        lstm = AugmentedLstm(10, 11)
        output = lstm(tensor)
        output_sequence, _ = pad_packed_sequence(output, batch_first=True)

        numpy.testing.assert_array_equal(output_sequence.data[1, 6:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence.data[2, 4:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence.data[3, 3:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence.data[4, 2:, :].numpy(), 0.0)

    def test_variable_length_sequences_run_backward_return_correctly_padded_outputs(self):
        sorted_tensor, sorted_sequence, _ = sort_batch_by_length(self.random_tensor, self.sequence_lengths)
        tensor = pack_padded_sequence(sorted_tensor, sorted_sequence.tolist(), batch_first=True)
        lstm = AugmentedLstm(10, 11, direction="backward")
        output = lstm(tensor)
        output_sequence, _ = pad_packed_sequence(output, batch_first=True)

        numpy.testing.assert_array_equal(output_sequence.data[1, 6:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence.data[2, 4:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence.data[3, 3:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence.data[4, 2:, :].numpy(), 0.0)

    def test_augmented_lstm_computes_same_function_as_pytorch_lstm(self):
        augmented_lstm = AugmentedLstm(10, 11)
        pytorch_lstm = LSTM(10, 11, num_layers=1, batch_first=True)
        # Initialize all weights to be == 1.
        initializer = InitializerApplicator(default_initializer=lambda tensor: torch.nn.init.constant(tensor, 1.))
        initializer(augmented_lstm)
        initializer(pytorch_lstm)

        initial_state = torch.autograd.Variable(torch.zeros([5, 11]))
        initial_memory = torch.autograd.Variable(torch.zeros([5, 11]))
        initial_state_pytorch = torch.autograd.Variable(torch.zeros([1, 5, 11]))
        initial_memory_pytorch = torch.autograd.Variable(torch.zeros([1, 5, 11]))

        # Use bigger numbers to avoid floating point instability.
        sorted_tensor, sorted_sequence, _ = sort_batch_by_length(self.random_tensor * 5., self.sequence_lengths)
        lstm_input = pack_padded_sequence(sorted_tensor, sorted_sequence.tolist(), batch_first=True)

        augmented_output = augmented_lstm(lstm_input, (initial_state, initial_memory))
        pytorch_output, _ = pytorch_lstm(lstm_input, (initial_state_pytorch, initial_memory_pytorch))
        pytorch_output_sequence, _ = pad_packed_sequence(pytorch_output, batch_first=True)
        augmented_output_sequence, _ = pad_packed_sequence(augmented_output, batch_first=True)

        numpy.testing.assert_array_almost_equal(pytorch_output_sequence.data.numpy(),
                                                augmented_output_sequence.data.numpy(), decimal=4)

    def test_augmented_lstm_works_with_highway_connections(self):
        augmented_lstm = AugmentedLstm(10, 11, use_highway=True)
        sorted_tensor, sorted_sequence, _ = sort_batch_by_length(self.random_tensor, self.sequence_lengths)
        lstm_input = pack_padded_sequence(sorted_tensor, sorted_sequence.tolist(), batch_first=True)
        augmented_lstm(lstm_input)

    def test_augmented_lstm_throws_error_on_non_packed_sequence_input(self):
        lstm = AugmentedLstm(3, 5)
        tensor = torch.rand([5, 7, 9])
        with pytest.raises(ConfigurationError):
            lstm(tensor)
