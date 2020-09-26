from typing import Any, Dict, List, Tuple, Union
import math
from collections import Counter

import torch
from torch.testing import assert_allclose

from allennlp.common.testing import (
    AllenNlpTestCase,
    multi_device,
    global_distributed_metric,
    run_distributed_test,
)
from allennlp.training.metrics import BLEU
from allennlp.training.util import ngrams, get_valid_tokens_mask


class BleuTest(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.metric = BLEU(ngram_weights=(0.5, 0.5), exclude_indices={0})

    @multi_device
    def test_get_valid_tokens_mask(self, device: str):
        tensor = torch.tensor([[1, 2, 3, 0], [0, 1, 1, 0]], device=device)
        result = get_valid_tokens_mask(tensor, self.metric._exclude_indices).long()
        check = torch.tensor([[1, 1, 1, 0], [0, 1, 1, 0]], device=device)
        assert_allclose(result, check)

    @multi_device
    def test_ngrams(self, device: str):
        tensor = torch.tensor([1, 2, 3, 1, 2, 0], device=device)

        exclude_indices = self.metric._exclude_indices

        # Unigrams.
        counts: Counter = Counter(ngrams(tensor, 1, exclude_indices))
        unigram_check = {(1,): 2, (2,): 2, (3,): 1}
        assert counts == unigram_check

        # Bigrams.
        counts = Counter(ngrams(tensor, 2, exclude_indices))
        bigram_check = {(1, 2): 2, (2, 3): 1, (3, 1): 1}
        assert counts == bigram_check

        # Trigrams.
        counts = Counter(ngrams(tensor, 3, exclude_indices))
        trigram_check = {(1, 2, 3): 1, (2, 3, 1): 1, (3, 1, 2): 1}
        assert counts == trigram_check

        # ngram size too big, no ngrams produced.
        counts = Counter(ngrams(tensor, 7, exclude_indices))
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

    def test_distributed_bleu(self):
        predictions = [
            torch.tensor([[1, 0, 0], [1, 1, 0]]),
            torch.tensor([[1, 1, 1]]),
        ]
        gold_targets = [
            torch.tensor([[2, 0, 0], [1, 0, 0]]),
            torch.tensor([[1, 1, 2]]),
        ]

        check = math.exp(0.5 * (math.log(3) - math.log(6)) + 0.5 * (math.log(1) - math.log(3)))
        metric_kwargs = {"predictions": predictions, "gold_targets": gold_targets}
        desired_values = {"BLEU": check}
        run_distributed_test(
            [-1, -1],
            global_distributed_metric,
            BLEU(ngram_weights=(0.5, 0.5), exclude_indices={0}),
            metric_kwargs,
            desired_values,
            exact=False,
        )

    def test_multiple_distributed_runs(self):
        predictions = [
            torch.tensor([[1, 0, 0], [1, 1, 0]]),
            torch.tensor([[1, 1, 1]]),
        ]
        gold_targets = [
            torch.tensor([[2, 0, 0], [1, 0, 0]]),
            torch.tensor([[1, 1, 2]]),
        ]

        check = math.exp(0.5 * (math.log(3) - math.log(6)) + 0.5 * (math.log(1) - math.log(3)))
        metric_kwargs = {"predictions": predictions, "gold_targets": gold_targets}
        desired_values = {"BLEU": check}
        run_distributed_test(
            [-1, -1],
            multiple_runs,
            BLEU(ngram_weights=(0.5, 0.5), exclude_indices={0}),
            metric_kwargs,
            desired_values,
            exact=False,
        )


def multiple_runs(
    global_rank: int,
    world_size: int,
    gpu_id: Union[int, torch.device],
    metric: BLEU,
    metric_kwargs: Dict[str, List[Any]],
    desired_values: Dict[str, Any],
    exact: Union[bool, Tuple[float, float]] = True,
):

    kwargs = {}
    # Use the arguments meant for the process with rank `global_rank`.
    for argname in metric_kwargs:
        kwargs[argname] = metric_kwargs[argname][global_rank]

    for i in range(200):
        metric(**kwargs)

    assert_allclose(desired_values["BLEU"], metric.get_metric()["BLEU"])
