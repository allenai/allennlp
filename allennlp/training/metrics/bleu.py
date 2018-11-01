from collections import Counter
import math
from typing import Iterable, Tuple, Dict, Set

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric


@Metric.register("bleu")
class BLEU(Metric):
    """
    Bilingual Evaluation Understudy (BLEU).

    BLEU is a common metric used for evaluating the quality of machine translations
    against a set of reference translations. See Papineni et. al.,
    "BLEU: a method for automatic evaluation of machine translation", 2002.

    Parameters
    ----------
    ngram_weights : ``Iterable[float]``, optional (default = (0.25, 0.25, 0.25, 0.25))
        Weights to assign to scores for each ngram size.

    Notes
    -----
    This implementation only considers a reference set of size 1, i.e. a single
    gold target sequence for each predicted sequence.
    """

    def __init__(self,
                 ngram_weights: Iterable[float] = (0.25, 0.25, 0.25, 0.25)) -> None:
        self._ngram_weights = ngram_weights
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

    @staticmethod
    def _ngrams(tensor: torch.LongTensor,
                ngram_size: int,
                exclude_indices: Set[int] = None) -> Dict[Tuple[int, ...], int]:
        ngram_counts: Dict[Tuple[int, ...], int] = Counter()
        if ngram_size > tensor.size(-1):
            return ngram_counts
        for start_position in range(ngram_size):
            for tensor_slice in tensor[start_position:].split(ngram_size, dim=-1):
                if tensor_slice.size(-1) < ngram_size:
                    break
                ngram = tuple(x.item() for x in tensor_slice)
                if exclude_indices and any(x in ngram for x in exclude_indices):
                    continue
                ngram_counts[ngram] += 1
        return ngram_counts

    def _get_modified_precision(self,
                                predicted_tokens: torch.LongTensor,
                                reference_tokens: torch.LongTensor,
                                ngram_size: int,
                                exclude_indices: Set[int] = None) -> Tuple[int, int]:
        clipped_matches = 0
        total_predicted = 0
        for batch_num in range(predicted_tokens.size(0)):
            predicted_row = predicted_tokens[batch_num, :]
            reference_row = reference_tokens[batch_num, :]
            predicted_ngram_counts = self._ngrams(predicted_row, ngram_size, exclude_indices)
            reference_ngram_counts = self._ngrams(reference_row, ngram_size, exclude_indices)
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

    @staticmethod
    def _get_valid_tokens_mask(tensor: torch.LongTensor, exclude_indices: Set[int]) -> torch.ByteTensor:
        valid_tokens_mask = torch.ones(tensor.size(), dtype=torch.uint8)
        for index in exclude_indices:
            valid_tokens_mask = valid_tokens_mask & (tensor != index)
        return valid_tokens_mask

    @overrides
    def __call__(self,
                 predictions: torch.LongTensor,
                 gold_targets: torch.LongTensor,
                 exclude_indices: Set[int] = None) -> None:
        """
        Parameters
        ----------
        predictions : ``torch.LongTensor``, required
            Batched predicted tokens of shape `(batch_size, max_sequence_length)`.
        references : ``torch.LongTensor``, required
            Batched reference (gold) translations with shape `(batch_size, max_gold_sequence_length)`.
        exclude_indices : ``Set[int]``, optional (default = None)
            Indices to exclude when calculating ngrams. This should usually include
            the indices of the start, end, and pad tokens.

        Returns
        -------
        None
        """
        predictions, gold_targets = self.unwrap_to_tensors(predictions, gold_targets)
        for ngram_size, _ in enumerate(self._ngram_weights, start=1):
            precision_matches, precision_totals = self._get_modified_precision(
                    predictions, gold_targets, ngram_size, exclude_indices)
            self._precision_matches[ngram_size] += precision_matches
            self._precision_totals[ngram_size] += precision_totals
        if not exclude_indices:
            self._prediction_lengths += predictions.size(0) * predictions.size(1)
            self._reference_lengths += gold_targets.size(0) * gold_targets.size(1)
        else:
            valid_predictions_mask = self._get_valid_tokens_mask(predictions, exclude_indices)
            self._prediction_lengths += valid_predictions_mask.sum().item()
            valid_gold_targets_mask = self._get_valid_tokens_mask(gold_targets, exclude_indices)
            self._reference_lengths += valid_gold_targets_mask.sum().item()

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        brevity_penalty = self._get_brevity_penalty()
        ngram_scores = (weight * (math.log(self._precision_matches[n] + 1e-13) -
                                  math.log(self._precision_totals[n] + 1e-13))
                        for n, weight in enumerate(self._ngram_weights, start=1))
        bleu = brevity_penalty * math.exp(sum(ngram_scores))
        if reset:
            self.reset()
        return {"BLEU": bleu}
