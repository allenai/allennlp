import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.seq2seq_encoders import QaNetEncoder
from allennlp.common.params import Params


class QaNetEncoderTest(AllenNlpTestCase):
    def test_qanet_encoder_can_build_from_params(self):
        params = Params(
            {
                "input_dim": 16,
                "hidden_dim": 16,
                "attention_projection_dim": 16,
                "feedforward_hidden_dim": 16,
                "num_blocks": 2,
                "num_convs_per_block": 2,
                "conv_kernel_size": 3,
                "num_attention_heads": 4,
                "dropout_prob": 0.1,
                "layer_dropout_undecayed_prob": 0.1,
                "attention_dropout_prob": 0,
            }
        )

        encoder = QaNetEncoder.from_params(params)
        assert isinstance(encoder, QaNetEncoder)
        assert encoder.get_input_dim() == 16
        assert encoder.get_output_dim() == 16

    def test_qanet_encoder_runs_forward(self):
        encoder = QaNetEncoder(
            input_dim=16,
            hidden_dim=16,
            attention_projection_dim=16,
            feedforward_hidden_dim=16,
            num_blocks=2,
            num_convs_per_block=2,
            conv_kernel_size=3,
            num_attention_heads=4,
            dropout_prob=0.1,
            layer_dropout_undecayed_prob=0.1,
            attention_dropout_prob=0.1,
        )
        inputs = torch.randn(2, 12, 16)
        assert list(encoder(inputs).size()) == [2, 12, 16]
