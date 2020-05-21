import numpy
import pytest
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from allennlp.modules.stacked_bidirectional_lstm import StackedBidirectionalLstm
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.common.params import Params
from allennlp.nn import InitializerApplicator, Initializer
from allennlp.nn.util import sort_batch_by_length


class TestStackedBidirectionalLstm:
    def test_stacked_bidirectional_lstm_completes_forward_pass(self):
        input_tensor = torch.rand(4, 5, 3)
        input_tensor[1, 4:, :] = 0.0
        input_tensor[2, 2:, :] = 0.0
        input_tensor[3, 1:, :] = 0.0
        input_tensor = pack_padded_sequence(input_tensor, [5, 4, 2, 1], batch_first=True)
        lstm = StackedBidirectionalLstm(3, 7, 3)
        output, _ = lstm(input_tensor)
        output_sequence, _ = pad_packed_sequence(output, batch_first=True)
        numpy.testing.assert_array_equal(output_sequence.data[1, 4:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence.data[2, 2:, :].numpy(), 0.0)
        numpy.testing.assert_array_equal(output_sequence.data[3, 1:, :].numpy(), 0.0)

    def test_stacked_bidirectional_lstm_can_build_from_params(self):
        params = Params(
            {
                "type": "stacked_bidirectional_lstm",
                "input_size": 5,
                "hidden_size": 9,
                "num_layers": 3,
            }
        )
        encoder = Seq2SeqEncoder.from_params(params)

        assert encoder.get_input_dim() == 5
        assert encoder.get_output_dim() == 18
        assert encoder.is_bidirectional

    def test_stacked_bidirectional_lstm_can_build_from_params_seq2vec(self):
        params = Params(
            {
                "type": "stacked_bidirectional_lstm",
                "input_size": 5,
                "hidden_size": 9,
                "num_layers": 3,
            }
        )
        encoder = Seq2VecEncoder.from_params(params)

        assert encoder.get_input_dim() == 5
        assert encoder.get_output_dim() == 18

    def test_stacked_bidirectional_lstm_can_complete_forward_pass_seq2vec(self):
        params = Params(
            {
                "type": "stacked_bidirectional_lstm",
                "input_size": 3,
                "hidden_size": 9,
                "num_layers": 3,
            }
        )
        encoder = Seq2VecEncoder.from_params(params)
        input_tensor = torch.rand(4, 5, 3)
        mask = torch.ones(4, 5).bool()
        output = encoder(input_tensor, mask)
        assert output.detach().numpy().shape == (4, 18)

    @pytest.mark.parametrize(
        "dropout_name", ("layer_dropout_probability", "recurrent_dropout_probability")
    )
    def test_stacked_bidirectional_lstm_dropout_version_is_different(self, dropout_name: str):
        stacked_lstm = StackedBidirectionalLstm(input_size=10, hidden_size=11, num_layers=3)
        if dropout_name == "layer_dropout_probability":
            dropped_stacked_lstm = StackedBidirectionalLstm(
                input_size=10, hidden_size=11, num_layers=3, layer_dropout_probability=0.9
            )
        elif dropout_name == "recurrent_dropout_probability":
            dropped_stacked_lstm = StackedBidirectionalLstm(
                input_size=10, hidden_size=11, num_layers=3, recurrent_dropout_probability=0.9
            )
        else:
            raise ValueError("Do not recognise the following dropout name " f"{dropout_name}")
        # Initialize all weights to be == 1.
        constant_init = Initializer.from_params(Params({"type": "constant", "val": 0.5}))
        initializer = InitializerApplicator([(".*", constant_init)])
        initializer(stacked_lstm)
        initializer(dropped_stacked_lstm)

        initial_state = torch.randn([3, 5, 11])
        initial_memory = torch.randn([3, 5, 11])

        tensor = torch.rand([5, 7, 10])
        sequence_lengths = torch.LongTensor([7, 7, 7, 7, 7])

        sorted_tensor, sorted_sequence, _, _ = sort_batch_by_length(tensor, sequence_lengths)
        lstm_input = pack_padded_sequence(
            sorted_tensor, sorted_sequence.data.tolist(), batch_first=True
        )

        stacked_output, stacked_state = stacked_lstm(lstm_input, (initial_state, initial_memory))
        dropped_output, dropped_state = dropped_stacked_lstm(
            lstm_input, (initial_state, initial_memory)
        )
        dropped_output_sequence, _ = pad_packed_sequence(dropped_output, batch_first=True)
        stacked_output_sequence, _ = pad_packed_sequence(stacked_output, batch_first=True)
        if dropout_name == "layer_dropout_probability":
            with pytest.raises(AssertionError):
                numpy.testing.assert_array_almost_equal(
                    dropped_output_sequence.data.numpy(),
                    stacked_output_sequence.data.numpy(),
                    decimal=4,
                )
        if dropout_name == "recurrent_dropout_probability":
            with pytest.raises(AssertionError):
                numpy.testing.assert_array_almost_equal(
                    dropped_state[0].data.numpy(), stacked_state[0].data.numpy(), decimal=4
                )
            with pytest.raises(AssertionError):
                numpy.testing.assert_array_almost_equal(
                    dropped_state[1].data.numpy(), stacked_state[1].data.numpy(), decimal=4
                )
