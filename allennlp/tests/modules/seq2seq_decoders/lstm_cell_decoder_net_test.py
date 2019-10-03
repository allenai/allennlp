import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.seq2seq_decoders import LstmCellDecoderNet
from allennlp.modules.attention import DotProductAttention


class TestLstmCellDecoderNet(AllenNlpTestCase):
    def test_lstm_cell_decoder_net_init(self):
        decoder_inout_dim = 10
        lstm_decoder_net = LstmCellDecoderNet(
            decoding_dim=decoder_inout_dim,
            target_embedding_dim=decoder_inout_dim,
            attention=DotProductAttention(),
            bidirectional_input=False,
        )
        batch_size = 5
        time_steps = 10
        encoded_state = torch.rand(batch_size, time_steps, decoder_inout_dim)
        source_mask = torch.ones(batch_size, time_steps)
        source_mask[0, 7:] = 0
        source_mask[1, 5:] = 0
        encoder_out = {"source_mask": source_mask, "encoder_outputs": encoded_state}
        decoder_init_state = lstm_decoder_net.init_decoder_state(encoder_out)
        assert list(decoder_init_state["decoder_hidden"].shape) == [batch_size, decoder_inout_dim]
        assert list(decoder_init_state["decoder_context"].shape) == [batch_size, decoder_inout_dim]

    def test_lstm_cell_decoder_net_forward(self):
        decoder_inout_dim = 10
        lstm_decoder_net = LstmCellDecoderNet(
            decoding_dim=decoder_inout_dim,
            target_embedding_dim=decoder_inout_dim,
            attention=DotProductAttention(),
            bidirectional_input=True,
        )
        batch_size = 5
        time_steps = 10
        encoded_state = torch.rand(batch_size, time_steps, decoder_inout_dim)
        source_mask = torch.ones(batch_size, time_steps)
        source_mask[0, 7:] = 0
        source_mask[1, 5:] = 0
        encoder_out = {"source_mask": source_mask, "encoder_outputs": encoded_state}
        prev_step_prediction_embeded = torch.rand(batch_size, 1, decoder_inout_dim)
        prev_state = lstm_decoder_net.init_decoder_state(encoder_out)

        next_state, decoded_vec = lstm_decoder_net(
            prev_state, encoded_state, source_mask, prev_step_prediction_embeded
        )
        assert list(next_state["decoder_hidden"].shape) == [batch_size, decoder_inout_dim]
        assert list(next_state["decoder_context"].shape) == [batch_size, decoder_inout_dim]
        assert list(decoded_vec.shape) == [batch_size, decoder_inout_dim]

    def test_lstm_cell_decoder_net_forward_without_attention(self):
        decoder_inout_dim = 10
        lstm_decoder_net = LstmCellDecoderNet(
            decoding_dim=decoder_inout_dim,
            target_embedding_dim=decoder_inout_dim,
            attention=None,
            bidirectional_input=True,
        )
        batch_size = 5
        time_steps = 10
        encoded_state = torch.rand(batch_size, time_steps, decoder_inout_dim)
        source_mask = torch.ones(batch_size, time_steps)
        source_mask[0, 7:] = 0
        source_mask[1, 5:] = 0
        encoder_out = {"source_mask": source_mask, "encoder_outputs": encoded_state}
        prev_step_prediction_embeded = torch.rand(batch_size, 1, decoder_inout_dim)
        prev_state = lstm_decoder_net.init_decoder_state(encoder_out)

        next_state, decoded_vec = lstm_decoder_net(
            prev_state, encoded_state, source_mask, prev_step_prediction_embeded
        )
        assert list(next_state["decoder_hidden"].shape) == [batch_size, decoder_inout_dim]
        assert list(next_state["decoder_context"].shape) == [batch_size, decoder_inout_dim]
        assert list(decoded_vec.shape) == [batch_size, decoder_inout_dim]

    def test_lstm_cell_decoder_net_forward_without_bidirectionality(self):
        decoder_inout_dim = 10
        lstm_decoder_net = LstmCellDecoderNet(
            decoding_dim=decoder_inout_dim,
            target_embedding_dim=decoder_inout_dim,
            attention=DotProductAttention(),
            bidirectional_input=False,
        )
        batch_size = 5
        time_steps = 10
        encoded_state = torch.rand(batch_size, time_steps, decoder_inout_dim)
        source_mask = torch.ones(batch_size, time_steps)
        source_mask[0, 7:] = 0
        source_mask[1, 5:] = 0
        encoder_out = {"source_mask": source_mask, "encoder_outputs": encoded_state}
        prev_step_prediction_embeded = torch.rand(batch_size, 1, decoder_inout_dim)
        prev_state = lstm_decoder_net.init_decoder_state(encoder_out)

        next_state, decoded_vec = lstm_decoder_net(
            prev_state, encoded_state, source_mask, prev_step_prediction_embeded
        )
        assert list(next_state["decoder_hidden"].shape) == [batch_size, decoder_inout_dim]
        assert list(next_state["decoder_context"].shape) == [batch_size, decoder_inout_dim]
        assert list(decoded_vec.shape) == [batch_size, decoder_inout_dim]
