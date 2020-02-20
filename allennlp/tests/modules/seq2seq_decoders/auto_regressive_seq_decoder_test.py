from typing import Any, Iterable, Dict

import pytest
import torch
from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import END_SYMBOL, prepare_environment, START_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Embedding
from allennlp.modules.seq2seq_decoders import AutoRegressiveSeqDecoder
from allennlp.modules.seq2seq_decoders import StackedSelfAttentionDecoderNet
from allennlp.training.metrics import BLEU, Metric


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


class DummyMetric(Metric):
    def __init__(self) -> None:
        self.reset()

    @staticmethod
    def f1(predicted: Iterable[Any], expected: Iterable[Any]) -> float:
        expected = frozenset(expected)
        predicted = frozenset(predicted)
        if len(predicted) <= 0 and len(expected) <= 0:
            return 1.0
        if len(predicted) <= 0 or len(expected) <= 0:
            return 0.0

        true_positive_count = len(predicted & expected)
        p = true_positive_count / len(predicted)
        r = true_positive_count / len(expected)
        return (2 * p * r) / (p + r)

    @overrides
    def __call__(self, best_span_strings, answer_strings):
        for best_span_string, answer_string in zip(best_span_strings, answer_strings):
            self._total_em += best_span_string == answer_string
            self._total_f1 += self.f1(best_span_string, answer_string)
            self._count += 1

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return {"em": exact_match, "f1": f1_score}

    @overrides
    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    def __str__(self):
        return f"DummyMetric(em={self._total_em}, f1={self._total_f1})"


class TestAutoRegressiveSeqDecoder(AllenNlpTestCase):
    def test_auto_regressive_seq_decoder_init(self):
        decoder_inout_dim = 4
        vocab, decoder_net = create_vocab_and_decoder_net(decoder_inout_dim)

        AutoRegressiveSeqDecoder(
            vocab,
            decoder_net,
            10,
            Embedding(num_embeddings=vocab.get_vocab_size(), embedding_dim=decoder_inout_dim),
        )

        with pytest.raises(ConfigurationError):
            AutoRegressiveSeqDecoder(
                vocab,
                decoder_net,
                10,
                Embedding(
                    num_embeddings=vocab.get_vocab_size(), embedding_dim=decoder_inout_dim + 1
                ),
            )

    def test_auto_regressive_seq_decoder_forward(self):
        batch_size, time_steps, decoder_inout_dim = 2, 3, 4
        vocab, decoder_net = create_vocab_and_decoder_net(decoder_inout_dim)

        auto_regressive_seq_decoder = AutoRegressiveSeqDecoder(
            vocab,
            decoder_net,
            10,
            Embedding(num_embeddings=vocab.get_vocab_size(), embedding_dim=decoder_inout_dim),
        )

        encoded_state = torch.rand(batch_size, time_steps, decoder_inout_dim)
        source_mask = torch.ones(batch_size, time_steps).long()
        target_tokens = {"tokens": {"tokens": torch.ones(batch_size, time_steps).long()}}
        source_mask[0, 1:] = 0
        encoder_out = {"source_mask": source_mask, "encoder_outputs": encoded_state}

        assert auto_regressive_seq_decoder.forward(encoder_out) == {}
        loss = auto_regressive_seq_decoder.forward(encoder_out, target_tokens)["loss"]
        assert loss.shape == torch.Size([]) and loss.requires_grad
        auto_regressive_seq_decoder.eval()
        assert "predictions" in auto_regressive_seq_decoder.forward(encoder_out)

    def test_auto_regressive_seq_decoder_indices_to_tokens(self):
        decoder_inout_dim = 4
        vocab, decoder_net = create_vocab_and_decoder_net(decoder_inout_dim)

        auto_regressive_seq_decoder = AutoRegressiveSeqDecoder(
            vocab,
            decoder_net,
            10,
            Embedding(num_embeddings=vocab.get_vocab_size(), embedding_dim=decoder_inout_dim),
        )

        predictions = torch.tensor([[3, 2, 5, 0, 0], [2, 2, 3, 5, 0]])

        tokens_ground_truth = [["B", "A"], ["A", "A", "B"]]
        predicted_tokens = auto_regressive_seq_decoder.indices_to_tokens(predictions.numpy())
        assert predicted_tokens == tokens_ground_truth

    def test_auto_regressive_seq_decoder_post_process(self):
        decoder_inout_dim = 4
        vocab, decoder_net = create_vocab_and_decoder_net(decoder_inout_dim)

        auto_regressive_seq_decoder = AutoRegressiveSeqDecoder(
            vocab,
            decoder_net,
            10,
            Embedding(num_embeddings=vocab.get_vocab_size(), embedding_dim=decoder_inout_dim),
        )

        predictions = torch.tensor([[3, 2, 5, 0, 0], [2, 2, 3, 5, 0]])

        tokens_ground_truth = [["B", "A"], ["A", "A", "B"]]

        output_dict = {"predictions": predictions}
        predicted_tokens = auto_regressive_seq_decoder.post_process(output_dict)["predicted_tokens"]
        assert predicted_tokens == tokens_ground_truth

    def test_auto_regressive_seq_decoder_tensor_and_token_based_metric(self):
        # set all seeds to a fixed value (torch, numpy, etc.).
        # this enable a deterministic behavior of the `auto_regressive_seq_decoder`
        # below (i.e., parameter initialization and `encoded_state = torch.randn(..)`)
        prepare_environment(Params({}))

        batch_size, time_steps, decoder_inout_dim = 2, 3, 4
        vocab, decoder_net = create_vocab_and_decoder_net(decoder_inout_dim)

        auto_regressive_seq_decoder = AutoRegressiveSeqDecoder(
            vocab,
            decoder_net,
            10,
            Embedding(num_embeddings=vocab.get_vocab_size(), embedding_dim=decoder_inout_dim),
            tensor_based_metric=BLEU(),
            token_based_metric=DummyMetric(),
        ).eval()

        encoded_state = torch.randn(batch_size, time_steps, decoder_inout_dim)
        source_mask = torch.ones(batch_size, time_steps).long()
        target_tokens = {"tokens": {"tokens": torch.ones(batch_size, time_steps).long()}}
        source_mask[0, 1:] = 0
        encoder_out = {"source_mask": source_mask, "encoder_outputs": encoded_state}

        auto_regressive_seq_decoder.forward(encoder_out, target_tokens)
        assert auto_regressive_seq_decoder.get_metrics()["BLEU"] == 1.388809517005903e-11
        assert auto_regressive_seq_decoder.get_metrics()["em"] == 0.0
        assert auto_regressive_seq_decoder.get_metrics()["f1"] == 1 / 3
