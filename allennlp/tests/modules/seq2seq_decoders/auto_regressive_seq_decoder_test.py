import torch
import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Embedding
from allennlp.modules.seq2seq_decoders import AutoRegressiveSeqDecoder
from allennlp.modules.seq2seq_decoders import StackedSelfAttentionDecoderNet
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL


def create_vocab_and_decoder_net(decoder_inout_dim):
    vocab = Vocabulary()
    vocab.add_tokens_to_namespace(["A", "B", START_SYMBOL, END_SYMBOL])

    decoder_net = StackedSelfAttentionDecoderNet(
        decoding_dim=decoder_inout_dim,
        target_embedding_dim=decoder_inout_dim,
        feedforward_hidden_dim=20,
        num_layers=2,
        num_attention_heads=4,
    )

    return vocab, decoder_net


class TestAutoRegressiveSeqDecoder(AllenNlpTestCase):
    def test_auto_regressive_seq_decoder_init(self):
        decoder_inout_dim = 4
        vocab, decoder_net = create_vocab_and_decoder_net(decoder_inout_dim)

        auto_regressive_seq_decoder = AutoRegressiveSeqDecoder(
            vocab, decoder_net, 10, Embedding(vocab.get_vocab_size(), decoder_inout_dim)
        )

        with pytest.raises(ConfigurationError):
            auto_regressive_seq_decoder = AutoRegressiveSeqDecoder(
                vocab, decoder_net, 10, Embedding(vocab.get_vocab_size(), decoder_inout_dim + 1)
            )

    def test_auto_regressive_seq_decoder_forward(self):
        batch_size, time_steps, decoder_inout_dim = 2, 3, 4
        vocab, decoder_net = create_vocab_and_decoder_net(decoder_inout_dim)

        auto_regressive_seq_decoder = AutoRegressiveSeqDecoder(
            vocab, decoder_net, 10, Embedding(vocab.get_vocab_size(), decoder_inout_dim)
        )

        encoded_state = torch.rand(batch_size, time_steps, decoder_inout_dim)
        source_mask = torch.ones(batch_size, time_steps).long()
        target_tokens = {"tokens": torch.ones(batch_size, time_steps).long()}
        source_mask[0, 1:] = 0
        encoder_out = {"source_mask": source_mask, "encoder_outputs": encoded_state}

        assert auto_regressive_seq_decoder.forward(encoder_out) == {}
        loss = auto_regressive_seq_decoder.forward(encoder_out, target_tokens)["loss"]
        assert loss.shape == torch.Size([]) and loss.requires_grad
        auto_regressive_seq_decoder.eval()
        assert "predictions" in auto_regressive_seq_decoder.forward(encoder_out)

    def test_auto_regressive_seq_decoder_indeces_to_tokens(self):
        decoder_inout_dim = 4
        vocab, decoder_net = create_vocab_and_decoder_net(decoder_inout_dim)

        auto_regressive_seq_decoder = AutoRegressiveSeqDecoder(
            vocab, decoder_net, 10, Embedding(vocab.get_vocab_size(), decoder_inout_dim)
        )

        predictions = torch.tensor([[3, 2, 5, 0, 0], [2, 2, 3, 5, 0]])

        tokens_ground_truth = [["B", "A"], ["A", "A", "B"]]
        predicted_tokens = auto_regressive_seq_decoder.indeces_to_tokens(predictions.numpy())
        assert predicted_tokens == tokens_ground_truth

    def test_auto_regressive_seq_decoder_post_process(self):
        decoder_inout_dim = 4
        vocab, decoder_net = create_vocab_and_decoder_net(decoder_inout_dim)

        auto_regressive_seq_decoder = AutoRegressiveSeqDecoder(
            vocab, decoder_net, 10, Embedding(vocab.get_vocab_size(), decoder_inout_dim)
        )

        predictions = torch.tensor([[3, 2, 5, 0, 0], [2, 2, 3, 5, 0]])

        tokens_ground_truth = [["B", "A"], ["A", "A", "B"]]

        output_dict = {"predictions": predictions}
        predicted_tokens = auto_regressive_seq_decoder.post_process(output_dict)["predicted_tokens"]
        assert predicted_tokens == tokens_ground_truth
