from collections import Counter
from typing import Tuple, Dict, Set

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric


# TODO: implement ROUGE-L


@Metric.register("rogue")
class ROUGE(Metric):
    """
    Recall-Oriented Understudy for Gisting Evaluation (ROUGE)

    ROUGE is a metric for measuring the quality of summaries. It is based on calculating the recall
    between ngrams in the predicted summary and a set of reference summaries. See [Lin,
    "ROUGE: A Package For Automatic Evaluation Of Summaries", 2004]
    (https://api.semanticscholar.org/CorpusID:964287).

    # Parameters

    ngram_size : `int`, optional (default = 2)
        ROUGE scores are calculate for ROUGE-1 .. ROUGE-`ngram_size`
    exclude_indices : `Set[int]`, optional (default = None)
        Indices to exclude when calculating ngrams. This should usually include
        the indices of the start, end, and pad tokens.
    """

    def __init__(self, ngram_size: int = 2, exclude_indices: Set[int] = None,) -> None:
        self._ngram_size = ngram_size
        self._exclude_indices = exclude_indices or set()
        self._recall_matches: Dict[int, int] = Counter()
        self._recall_totals: Dict[int, int] = Counter()

    @overrides
    def reset(self) -> None:
        self._recall_matches = Counter()
        self._recall_totals = Counter()

    def _get_recall_counts(
        self,
        predicted_tokens: torch.LongTensor,
        reference_tokens: torch.LongTensor,
        ngram_size: int,
    ) -> Tuple[int, int]:
        """
        Compare the predicted tokens to the reference (gold) tokens at the desired
        ngram size and calculate the numerator and denominator for recall.

        The numerator is the number of ngrams in the predicted sentences that match
        with an ngram in the corresponding reference sentence, clipped by the total
        count of that ngram in the reference sentence. The denominator is
        the total count of reference ngrams.
        """
        # TODO: fix not being able to import this normally
        from allennlp.training.util import ngrams

        matches = 0
        total_reference = 0
        for batch_num in range(predicted_tokens.size(0)):
            predicted_row = predicted_tokens[batch_num, :]
            reference_row = reference_tokens[batch_num, :]
            predicted_ngram_counts = ngrams(predicted_row, ngram_size, self._exclude_indices)
            reference_ngram_counts = ngrams(reference_row, ngram_size, self._exclude_indices)
            for ngram, count in reference_ngram_counts.items():
                matches += min(predicted_ngram_counts[ngram], count)
                total_reference += count
        return matches, total_reference

    @overrides
    def __call__(
        self,  # type: ignore
        predictions: torch.LongTensor,
        gold_targets: torch.LongTensor,
    ) -> None:
        """
        Update recall counts.

        # Parameters

        predictions : `torch.LongTensor`, required
            Batched predicted tokens of shape `(batch_size, max_sequence_length)`.
        references : `torch.LongTensor`, required
            Batched reference (gold) sequences with shape `(batch_size, max_gold_sequence_length)`.

        # Returns

        None
        """
        predictions, gold_targets = self.detach_tensors(predictions, gold_targets)
        for n in range(1, self._ngram_size + 1):
            recall_matches, recall_totals = self._get_recall_counts(predictions, gold_targets, n)
            self._recall_matches[n] += recall_matches
            self._recall_totals[n] += recall_totals

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        """
        # Parameters

        reset : `bool`, optional (default = False)
            Reset any accumulators or internal state.

        # Returns

        Dict[str, float]:
            A dictionary containing `ROUGE-1` .. `ROUGE-ngram_size` scores.
        """
        metrics = {
            f"ROUGE{i}": self._recall_matches[i] / self._recall_totals[i]
            for i in range(1, self._ngram_size + 1)
        }

        if reset:
            self.reset()

        return metrics
