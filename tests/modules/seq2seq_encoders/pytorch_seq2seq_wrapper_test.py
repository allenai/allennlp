# pylint: disable=no-self-use,invalid-name
from numpy.testing import assert_almost_equal
import pytest
import torch
from torch.autograd import Variable
from torch.nn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.common.tensor import sort_batch_by_length
from allennlp.testing.test_case import AllenNlpTestCase
from allennlp.common.checks import ConfigurationError

class TestPytorchSeq2SeqWrapper(AllenNlpTestCase):
    def test_get_output_dim_is_correct(self):
        lstm = LSTM(bidirectional=True, num_layers=3, input_size=2, hidden_size=7, batch_first=True)
        encoder = PytorchSeq2SeqWrapper(lstm)
        assert encoder.get_output_dim() == 14
        lstm = LSTM(bidirectional=False, num_layers=3, input_size=2, hidden_size=7, batch_first=True)
        encoder = PytorchSeq2SeqWrapper(lstm)
        assert encoder.get_output_dim() == 7

    def test_forward_pulls_out_correct_tensor_without_sequence_lengths(self):
        lstm = LSTM(bidirectional=True, num_layers=3, input_size=2, hidden_size=7, batch_first=True)
        encoder = PytorchSeq2SeqWrapper(lstm)
        input_tensor = Variable(torch.FloatTensor([[[.7, .8], [.1, 1.5]]]))
        lstm_output = lstm(input_tensor)
        encoder_output = encoder(input_tensor)
        assert_almost_equal(encoder_output.data.numpy(), lstm_output[0].data.numpy())

    def test_forward_pulls_out_correct_tensor_with_sequence_lengths(self):
        lstm = LSTM(bidirectional=True, num_layers=3, input_size=3, hidden_size=7, batch_first=True)
        encoder = PytorchSeq2SeqWrapper(lstm)
        tensor = torch.rand([5, 7, 3])
        tensor[1, 6:, :] = 0
        tensor[2, 4:, :] = 0
        tensor[3, 2:, :] = 0
        tensor[4, 1:, :] = 0
        input_tensor = Variable(tensor)
        sequence_lengths = torch.LongTensor([7, 6, 4, 2, 1])
        packed_sequence = pack_padded_sequence(input_tensor, list(sequence_lengths), batch_first=True)
        lstm_output, _ = lstm(packed_sequence)
        encoder_output = encoder(input_tensor, sequence_lengths)
        lstm_tensor, _ = pad_packed_sequence(lstm_output, batch_first=True)
        assert_almost_equal(encoder_output.data.numpy(), lstm_tensor.data.numpy())

    def test_forward_pulls_out_correct_tensor_for_unsorted_batches(self):
        lstm = LSTM(bidirectional=True, num_layers=3, input_size=3, hidden_size=7, batch_first=True)
        encoder = PytorchSeq2SeqWrapper(lstm)
        tensor = torch.rand([5, 7, 3])
        tensor[0, 3:, :] = 0
        tensor[1, 4:, :] = 0
        tensor[2, 2:, :] = 0
        tensor[3, 6:, :] = 0
        input_tensor = torch.autograd.Variable(tensor)
        sequence_lengths = torch.LongTensor([3, 4, 2, 6, 7])
        sorted_inputs, sorted_sequence_lengths, restoration_indices = sort_batch_by_length(input_tensor,
                                                                                           sequence_lengths)
        packed_sequence = pack_padded_sequence(sorted_inputs, list(sorted_sequence_lengths), batch_first=True)
        lstm_output, _ = lstm(packed_sequence)
        encoder_output = encoder(input_tensor, sequence_lengths)
        lstm_tensor, _ = pad_packed_sequence(lstm_output, batch_first=True)
        assert_almost_equal(encoder_output.data.numpy(), lstm_tensor[restoration_indices].data.numpy())

    def test_wrapper_raises_if_batch_first_is_false(self):

        with pytest.raises(ConfigurationError):
            lstm = LSTM(bidirectional=True, num_layers=3, input_size=3, hidden_size=7)
            _ = PytorchSeq2SeqWrapper(lstm)
