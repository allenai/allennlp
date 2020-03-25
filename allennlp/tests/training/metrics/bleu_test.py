import math
from collections import Counter

import torch
from torch.testing import assert_allclose

from allennlp.common.testing import AllenNlpTestCase, multi_device
from allennlp.training.metrics import BLEU


class BleuTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.metric = BLEU(ngram_weights=(0.5, 0.5), exclude_indices={0})

    @multi_device
    def test_get_valid_tokens_mask(self, device: str):
        tensor = torch.tensor([[1, 2, 3, 0], [0, 1, 1, 0]], device=device)
        result = self.metric._get_valid_tokens_mask(tensor).long()
        check = torch.tensor([[1, 1, 1, 0], [0, 1, 1, 0]], device=device)
        assert_allclose(result, check)

    @multi_device
    def test_ngrams(self, device: str):
        tensor = torch.tensor([1, 2, 3, 1, 2, 0], device=device)

        # Unigrams.
        counts: Counter = Counter(self.metric._ngrams(tensor, 1))
        unigram_check = {(1,): 2, (2,): 2, (3,): 1}
        assert counts == unigram_check

        # Bigrams.
        counts = Counter(self.metric._ngrams(tensor, 2))
        bigram_check = {(1, 2): 2, (2, 3): 1, (3, 1): 1}
        assert counts == bigram_check

        # Trigrams.
        counts = Counter(self.metric._ngrams(tensor, 3))
        trigram_check = {(1, 2, 3): 1, (2, 3, 1): 1, (3, 1, 2): 1}
        assert counts == trigram_check

        # ngram size too big, no ngrams produced.
        counts = Counter(self.metric._ngrams(tensor, 7))
        assert counts == {}

    @multi_device
    def test_bleu_computed_correctly(self, device: str):
        self.metric.reset()

        # shape: (batch_size, max_sequence_length)
        predictions = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], device=device)

        # shape: (batch_size, max_gold_sequence_length)
        gold_targets = torch.tensor([[2, 0, 0], [1, 0, 0], [1, 1, 2]], device=device)

        self.metric(predictions, gold_targets)

        assert self.metric._prediction_lengths == 6
        assert self.metric._reference_lengths == 5

        # Number of unigrams in predicted sentences that match gold sentences
        # (but not more than maximum occurrence of gold unigram within batch).
        assert self.metric._precision_matches[1] == (
            0
            + 1  # no matches in first sentence.
            + 2  # one clipped match in second sentence.  # two clipped matches in third sentence.
        )

        # Total number of predicted unigrams.
        assert self.metric._precision_totals[1] == (1 + 2 + 3)

        # Number of bigrams in predicted sentences that match gold sentences
        # (but not more than maximum occurrence of gold bigram within batch).
        assert self.metric._precision_matches[2] == (0 + 0 + 1)

        # Total number of predicted bigrams.
        assert self.metric._precision_totals[2] == (0 + 1 + 2)

        # Brevity penalty should be 1.0
        assert self.metric._get_brevity_penalty() == 1.0

        bleu = self.metric.get_metric(reset=True)["BLEU"]
        check = math.exp(0.5 * (math.log(3) - math.log(6)) + 0.5 * (math.log(1) - math.log(3)))
        assert_allclose(bleu, check)

    @multi_device
    def test_bleu_computed_with_zero_counts(self, device: str):
        self.metric.reset()
        assert self.metric.get_metric()["BLEU"] == 0
