# pylint: disable=invalid-name,no-self-use
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.seq2seq_encoders.bidirectional_language_model_transformer import \
        BidirectionalLanguageModelTransformer

class TestBidirectionalLanguageModelTransformer(AllenNlpTestCase):
    def test_bidirectional_transformer_encoder(self):
        transformer_encoder = BidirectionalLanguageModelTransformer(input_dim=32,
                                                                    hidden_dim=64,
                                                                    num_layers=2)
        token_embeddings = torch.rand(5, 10, 32)
        mask = torch.ones(5, 10)
        mask[0, 7:] = 0
        mask[1, 5:] = 0

        output = transformer_encoder(token_embeddings, mask)
        assert list(output.size()) == [5, 10, 64]

    def test_bidirectional_transfomer_all_layers(self):
        transformer_encoder = BidirectionalLanguageModelTransformer(input_dim=32,
                                                                    hidden_dim=64,
                                                                    num_layers=2,
                                                                    return_all_layers=True)
        token_embeddings = torch.rand(5, 10, 32)
        mask = torch.ones(5, 10)
        mask[0, 7:] = 0
        mask[1, 5:] = 0

        output = transformer_encoder(token_embeddings, mask)
        assert len(output) == 2

        concat_layers = torch.cat(
                [layer.unsqueeze(1) for layer in output], dim=1
        )

        # (batch_size, num_layers, timesteps, 2*input_dim)
        assert list(concat_layers.size()) == [5, 2, 10, 64]

    def test_attention_masks(self):
        transformer_encoder = BidirectionalLanguageModelTransformer(input_dim=32,
                                                                    hidden_dim=64,
                                                                    num_layers=2)

        mask = torch.ones(3, 6).int()
        mask[0, 3:] = 0
        mask[1, 5:] = 0

        forward_mask, backward_mask = transformer_encoder.get_attention_masks(mask)

        # rows = position in sequence
        # columns = positions used for attention
        assert (forward_mask[0].data == torch.IntTensor([[1, 0, 0, 0, 0, 0],
                                                         [1, 1, 0, 0, 0, 0],
                                                         [1, 1, 1, 0, 0, 0],
                                                         [0, 0, 0, 0, 0, 0],
                                                         [0, 0, 0, 0, 0, 0],
                                                         [0, 0, 0, 0, 0, 0]])).all()

        assert (forward_mask[1].data == torch.IntTensor([[1, 0, 0, 0, 0, 0],
                                                         [1, 1, 0, 0, 0, 0],
                                                         [1, 1, 1, 0, 0, 0],
                                                         [1, 1, 1, 1, 0, 0],
                                                         [1, 1, 1, 1, 1, 0],
                                                         [0, 0, 0, 0, 0, 0]])).all()

        assert (forward_mask[2].data == torch.IntTensor([[1, 0, 0, 0, 0, 0],
                                                         [1, 1, 0, 0, 0, 0],
                                                         [1, 1, 1, 0, 0, 0],
                                                         [1, 1, 1, 1, 0, 0],
                                                         [1, 1, 1, 1, 1, 0],
                                                         [1, 1, 1, 1, 1, 1]])).all()

        assert (backward_mask[0].data == torch.IntTensor([[1, 1, 1, 0, 0, 0],
                                                          [0, 1, 1, 0, 0, 0],
                                                          [0, 0, 1, 0, 0, 0],
                                                          [0, 0, 0, 0, 0, 0],
                                                          [0, 0, 0, 0, 0, 0],
                                                          [0, 0, 0, 0, 0, 0]])).all()

        assert (backward_mask[1].data == torch.IntTensor([[1, 1, 1, 1, 1, 0],
                                                          [0, 1, 1, 1, 1, 0],
                                                          [0, 0, 1, 1, 1, 0],
                                                          [0, 0, 0, 1, 1, 0],
                                                          [0, 0, 0, 0, 1, 0],
                                                          [0, 0, 0, 0, 0, 0]])).all()

        assert (backward_mask[2].data == torch.IntTensor([[1, 1, 1, 1, 1, 1],
                                                          [0, 1, 1, 1, 1, 1],
                                                          [0, 0, 1, 1, 1, 1],
                                                          [0, 0, 0, 1, 1, 1],
                                                          [0, 0, 0, 0, 1, 1],
                                                          [0, 0, 0, 0, 0, 1]])).all()
