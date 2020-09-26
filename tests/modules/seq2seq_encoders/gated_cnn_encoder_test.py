import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.seq2seq_encoders.gated_cnn_encoder import GatedCnnEncoder


class TestGatedCnnEncoder(AllenNlpTestCase):
    def test_gated_cnn_encoder(self):
        cnn_encoder = GatedCnnEncoder(
            input_dim=32,
            layers=[[[4, 32]], [[1, 16], [5, 16], [1, 32]], [[1, 64], [5, 64], [1, 32]]],
        )

        token_embeddings = torch.rand(5, 10, 32)
        mask = torch.ones(5, 10).bool()
        mask[0, 7:] = False
        mask[1, 5:] = False

        output = cnn_encoder(token_embeddings, mask)
        assert list(output.size()) == [5, 10, 64]

    def test_gated_cnn_encoder_dilations(self):
        cnn_encoder = GatedCnnEncoder(
            input_dim=32, layers=[[[2, 32, 1]], [[2, 32, 2]], [[2, 32, 4]], [[2, 32, 8]]]
        )

        token_embeddings = torch.rand(5, 10, 32)
        mask = torch.ones(5, 10).bool()
        mask[0, 7:] = False
        mask[1, 5:] = False

        output = cnn_encoder(token_embeddings, mask)
        assert list(output.size()) == [5, 10, 64]

    def test_gated_cnn_encoder_layers(self):
        cnn_encoder = GatedCnnEncoder(
            input_dim=32,
            layers=[[[4, 32]], [[1, 16], [5, 16], [1, 32]], [[1, 64], [5, 64], [1, 32]]],
            return_all_layers=True,
        )

        token_embeddings = torch.rand(5, 10, 32)
        mask = torch.ones(5, 10).bool()
        mask[0, 7:] = False
        mask[1, 5:] = False

        output = cnn_encoder(token_embeddings, mask)
        assert len(output) == 3
        concat_layers = torch.cat([layer.unsqueeze(1) for layer in output], dim=1)
        assert list(concat_layers.size()) == [5, 3, 10, 64]
