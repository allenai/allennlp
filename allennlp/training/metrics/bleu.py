from collections import Counter
import math
from typing import Iterable, Tuple, Dict, Set

from overrides import overrides
import torch
import torch.distributed as dist

from allennlp.common.util import is_distributed
from allennlp.training.metrics.metric import Metric
from allennlp.nn.util import dist_reduce_sum


@Metric.register("bleu")
class BLEU(Metric):
    """
    Bilingual Evaluation Understudy (BLEU).

    BLEU is a common metric used for evaluating the quality of machine translations
    against a set of reference translations. See
    [Papineni et. al., "BLEU: a method for automatic evaluation of machine translation", 2002][1].

    # Parameters

    ngram_weights : `Iterable[float]`, optional (default = `(0.25, 0.25, 0.25, 0.25)`)
        Weights to assign to scores for each ngram size.
    exclude_indices : `Set[int]`, optional (default = `None`)
        Indices to exclude when calculating ngrams. This should usually include
        the indices of the start, end, and pad tokens.

    # Notes

    We chose to implement this from scratch instead of wrapping an existing implementation
    (such as `nltk.translate.bleu_score`) for a two reasons. First, so that we could
    pass tensors directly to this metric instead of first converting the tensors to lists of strings.
    And second, because functions like `nltk.translate.bleu_score.corpus_bleu()` are
    meant to be called once over the entire corpus, whereas it is more efficient
    in our use case to update the running precision counts every batch.

    This implementation only considers a reference set of size 1, i.e. a single
    gold target sequence for each predicted sequence.


    [1]: https://www.semanticscholar.org/paper/8ff93cfd37dced279134c9d642337a2085b31f59/
    """

    def __init__(
        self,
        ngram_weights: Iterable[float] = (0.25, 0.25, 0.25, 0.25),
        exclude_indices: Set[int] = None,
    ) -> None:
        self._ngram_weights = ngram_weights
        self._exclude_indices = exclude_indices or set()
        self._precision_matches: Dict[int, int] = Counter()
        self._precision_totals: Dict[int, int] = Counter()
        self._prediction_lengths = 0
        self._reference_lengths = 0

    @overrides
    def reset(self) -> None:
        self._precision_matches = Counter()
        self._precision_totals = Counter()
        self._prediction_lengths = 0
        self._reference_lengths = 0

    def _get_modified_precision_counts(
        self,
        predicted_tokens: torch.LongTensor,
        reference_tokens: torch.LongTensor,
        ngram_size: int,
    ) -> Tuple[int, int]:
        """
        Compare the predicted tokens to the reference (gold) tokens at the desired
        ngram size and calculate the numerator and denominator for a modified
        form of precision.

        The numerator is the number of ngrams in the predicted sentences that match
        with an ngram in the corresponding reference sentence, clipped by the total
        count of that ngram in the reference sentence. The denominator is just
        the total count of predicted ngrams.
        """
        clipped_matches = 0
        total_predicted = 0
        from allennlp.training.util import ngrams

        for predicted_row, reference_row in zip(predicted_tokens, reference_tokens):
            predicted_ngram_counts = ngrams(predicted_row, ngram_size, self._exclude_indices)
            reference_ngram_counts = ngrams(reference_row, ngram_size, self._exclude_indices)
            for ngram, count in predicted_ngram_counts.items():
                clipped_matches += min(count, reference_ngram_counts[ngram])
                total_predicted += count
        return clipped_matches, total_predicted

    def _get_brevity_penalty(self) -> float:
        if self._prediction_lengths > self._reference_lengths:
            return 1.0
        if self._reference_lengths == 0 or self._prediction_lengths == 0:
            return 0.0
        return math.exp(1.0 - self._reference_lengths / self._prediction_lengths)

    @overrides
    def __call__(
        self,  # type: ignore
        predictions: torch.LongTensor,
        gold_targets: torch.LongTensor,
    ) -> None:
        """
        Update precision counts.

        # Parameters

        predictions : `torch.LongTensor`, required
            Batched predicted tokens of shape `(batch_size, max_sequence_length)`.
        references : `torch.LongTensor`, required
            Batched reference (gold) translations with shape `(batch_size, max_gold_sequence_length)`.

        # Returns

        None
        """
        predictions, gold_targets = self.detach_tensors(predictions, gold_targets)
        if is_distributed():
            world_size = dist.get_world_size()
        else:
            world_size = 1

        for ngram_size, _ in enumerate(self._ngram_weights, start=1):
            precision_matches, precision_totals = self._get_modified_precision_counts(
                predictions, gold_targets, ngram_size
            )

            self._precision_matches[ngram_size] += dist_reduce_sum(precision_matches) / world_size
            self._precision_totals[ngram_size] += dist_reduce_sum(precision_totals) / world_size

        if not self._exclude_indices:
            _prediction_lengths = predictions.size(0) * predictions.size(1)
            _reference_lengths = gold_targets.size(0) * gold_targets.size(1)

        else:
            from allennlp.training.util import get_valid_tokens_mask

            valid_predictions_mask = get_valid_tokens_mask(predictions, self._exclude_indices)
            valid_gold_targets_mask = get_valid_tokens_mask(gold_targets, self._exclude_indices)
            _prediction_lengths = valid_predictions_mask.sum().item()
            _reference_lengths = valid_gold_targets_mask.sum().item()

        self._prediction_lengths += dist_reduce_sum(_prediction_lengths)
        self._reference_lengths += dist_reduce_sum(_reference_lengths)

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:

        brevity_penalty = self._get_brevity_penalty()
        ngram_scores = (
            weight
            * (
                math.log(self._precision_matches[n] + 1e-13)
                - math.log(self._precision_totals[n] + 1e-13)
            )
            for n, weight in enumerate(self._ngram_weights, start=1)
        )
        bleu = brevity_penalty * math.exp(sum(ngram_scores))

        if reset:
            self.reset()
        return {"BLEU": bleu}
