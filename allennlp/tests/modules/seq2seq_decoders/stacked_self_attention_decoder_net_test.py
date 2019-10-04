import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.seq2seq_decoders import StackedSelfAttentionDecoderNet


class TestStackedSelfAttentionDecoderNet(AllenNlpTestCase):
    def test_stacked_self_attention_decoder_net_init(self):
        decoder_inout_dim = 10
        decoder_net = StackedSelfAttentionDecoderNet(
            decoding_dim=decoder_inout_dim,
            target_embedding_dim=decoder_inout_dim,
            feedforward_hidden_dim=20,
            num_layers=2,
            num_attention_heads=5,
        )
        batch_size = 5
        time_steps = 10
        encoded_state = torch.rand(batch_size, time_steps, decoder_inout_dim)
        source_mask = torch.ones(batch_size, time_steps)
        source_mask[0, 7:] = 0
        source_mask[1, 5:] = 0
        encoder_out = {"source_mask": source_mask, "encoder_outputs": encoded_state}
        decoder_init_state = decoder_net.init_decoder_state(encoder_out)
        assert decoder_init_state == {}

    def test_stacked_self_attention_decoder_net_forward(self):
        decoder_inout_dim = 10
        decoder_net = StackedSelfAttentionDecoderNet(
            decoding_dim=decoder_inout_dim,
            target_embedding_dim=decoder_inout_dim,
            feedforward_hidden_dim=20,
            num_layers=2,
            num_attention_heads=5,
        )
        batch_size = 5
        time_steps = 10
        encoded_state = torch.rand(batch_size, time_steps, decoder_inout_dim)
        source_mask = torch.ones(batch_size, time_steps)
        source_mask[0, 7:] = 0
        source_mask[1, 5:] = 0
        prev_timesteps = 3
        prev_step_prediction_embeded = torch.rand(batch_size, prev_timesteps, decoder_inout_dim)

        next_state, decoded_vec = decoder_net(
            {}, encoded_state, source_mask, prev_step_prediction_embeded
        )
        assert next_state == {}
        assert list(decoded_vec.shape) == [batch_size, prev_timesteps, decoder_inout_dim]
