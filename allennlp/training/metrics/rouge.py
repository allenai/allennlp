from collections import defaultdict
from typing import Tuple, Dict, Set

from overrides import overrides
import torch
import torch.distributed as dist

from allennlp.common.util import is_distributed
from allennlp.training.metrics.metric import Metric


@Metric.register("rouge")
class ROUGE(Metric):
    """
    Recall-Oriented Understudy for Gisting Evaluation (ROUGE)

    ROUGE is a metric for measuring the quality of summaries. It is based on calculating the recall
    between ngrams in the predicted summary and a set of reference summaries. See [Lin,
    "ROUGE: A Package For Automatic Evaluation Of Summaries", 2004]
    (https://api.semanticscholar.org/CorpusID:964287).

    # Parameters

    ngram_size : `int`, optional (default = `2`)
        ROUGE scores are calculate for ROUGE-1 .. ROUGE-`ngram_size`
    exclude_indices : `Set[int]`, optional (default = `None`)
        Indices to exclude when calculating ngrams. This should usually include
        the indices of the start, end, and pad tokens.
    """

    def __init__(
        self,
        ngram_size: int = 2,
        exclude_indices: Set[int] = None,
    ) -> None:
        self._ngram_size = ngram_size
        self._exclude_indices = exclude_indices or set()

        self._total_rouge_n_recalls: Dict[int, float] = defaultdict(lambda: 0.0)
        self._total_rouge_n_precisions: Dict[int, float] = defaultdict(lambda: 0.0)
        self._total_rouge_n_f1s: Dict[int, float] = defaultdict(lambda: 0.0)

        self._total_rouge_l_f1 = 0.0

        self._total_sequence_count = 0

    @overrides
    def reset(self) -> None:
        self._total_rouge_n_recalls = defaultdict(lambda: 0.0)
        self._total_rouge_n_precisions = defaultdict(lambda: 0.0)
        self._total_rouge_n_f1s = defaultdict(lambda: 0.0)

        self._total_rouge_l_f1 = 0.0

        self._total_sequence_count = 0

    def _longest_common_subsequence(self, seq_1: torch.LongTensor, seq_2: torch.LongTensor):
        """
        Computes the longest common subsequences between `seq_1` and `seq_2`, ignoring `self._exclude_indices`.
        """
        m = len(seq_1)
        n = len(seq_2)

        # Slightly lower memory usage by iterating over the longer sequence in outer loop
        # and storing previous lcs for the shorter sequence
        if m < n:
            seq_1, seq_2 = seq_2, seq_1
            m, n = n, m

        prev_lcs = torch.zeros(n + 1, dtype=torch.long)

        for i in range(m - 1, -1, -1):
            # Make sure we don't count special tokens as part of the subsequences
            if seq_1[i].item() in self._exclude_indices:
                continue

            cur_lcs = torch.zeros_like(prev_lcs)
            for j in range(n - 1, -1, -1):
                if seq_1[i] == seq_2[j]:
                    cur_lcs[j] = 1 + prev_lcs[j + 1]
                else:
                    cur_lcs[j] = max(cur_lcs[j + 1], prev_lcs[j])
            prev_lcs = cur_lcs

        return prev_lcs[0].item()

    def _get_rouge_l_score(
        self, predicted_tokens: torch.LongTensor, reference_tokens: torch.LongTensor
    ) -> float:
        """
        Compute sum of F1 scores given batch of predictions and references.
        """
        total_f1 = 0.0

        for predicted_seq, reference_seq in zip(predicted_tokens, reference_tokens):
            from allennlp.training.util import get_valid_tokens_mask

            m = get_valid_tokens_mask(reference_seq, self._exclude_indices).sum().item()
            n = get_valid_tokens_mask(predicted_seq, self._exclude_indices).sum().item()

            lcs = self._longest_common_subsequence(reference_seq, predicted_seq)

            # This also rules out the case that m or n are 0, so we don't worry about it later
            if lcs == 0:
                continue

            recall_lcs = lcs / m
            precision_lcs = lcs / n

            f1 = 2 * recall_lcs * precision_lcs / (recall_lcs + precision_lcs)

            total_f1 += f1

        if is_distributed():
            device = predicted_tokens.device
            _total_f1 = torch.tensor(total_f1).to(device)
            dist.all_reduce(_total_f1, op=dist.ReduceOp.SUM)
            total_f1 = _total_f1.item()

        return total_f1

    def _get_rouge_n_stats(
        self,
        predicted_tokens: torch.LongTensor,
        reference_tokens: torch.LongTensor,
        ngram_size: int,
    ) -> Tuple[float, float, float]:
        """
        Compare the predicted tokens to the reference (gold) tokens at the desired
        ngram size and compute recall, precision and f1 sums
        """
        total_recall = 0.0
        total_precision = 0.0
        total_f1 = 0.0

        for predicted_seq, reference_seq in zip(predicted_tokens, reference_tokens):
            from allennlp.training.util import ngrams

            predicted_ngram_counts = ngrams(predicted_seq, ngram_size, self._exclude_indices)
            reference_ngram_counts = ngrams(reference_seq, ngram_size, self._exclude_indices)

            matches = 0
            total_reference_ngrams = 0
            for ngram, count in reference_ngram_counts.items():
                matches += min(predicted_ngram_counts[ngram], count)
                total_reference_ngrams += count

            total_predicted_ngrams = sum(predicted_ngram_counts.values())

            if total_reference_ngrams == 0 or total_predicted_ngrams == 0 or matches == 0:
                continue

            recall = matches / total_reference_ngrams
            precision = matches / total_predicted_ngrams

            f1 = 2.0 * recall * precision / (recall + precision)

            # Accumulate stats
            total_recall += recall
            total_precision += precision
            total_f1 += f1

        if is_distributed():
            device = predicted_tokens.device
            _total_recall = torch.tensor(total_recall).to(device)
            _total_precision = torch.tensor(total_precision).to(device)
            _total_f1 = torch.tensor(total_f1).to(device)
            dist.all_reduce(_total_recall, op=dist.ReduceOp.SUM)
            dist.all_reduce(_total_precision, op=dist.ReduceOp.SUM)
            dist.all_reduce(_total_f1, op=dist.ReduceOp.SUM)
            total_recall = _total_recall.item()
            total_precision = _total_precision.item()
            total_f1 = _total_f1.item()

        return total_recall, total_precision, total_f1

    @overrides
    def __call__(
        self,  # type: ignore
        predictions: torch.LongTensor,
        gold_targets: torch.LongTensor,
    ) -> None:
        """
        Update recall counts.

        # Parameters

        predictions : `torch.LongTensor`
            Batched predicted tokens of shape `(batch_size, max_sequence_length)`.
        references : `torch.LongTensor`
            Batched reference (gold) sequences with shape `(batch_size, max_gold_sequence_length)`.

        # Returns

        None
        """
        # ROUGE-N
        predictions, gold_targets = self.detach_tensors(predictions, gold_targets)
        for n in range(1, self._ngram_size + 1):

            recall, precision, f1 = self._get_rouge_n_stats(predictions, gold_targets, n)
            self._total_rouge_n_recalls[n] += recall
            self._total_rouge_n_precisions[n] += precision
            self._total_rouge_n_f1s[n] += f1

        # ROUGE-L
        self._total_rouge_l_f1 += self._get_rouge_l_score(predictions, gold_targets)

        sequence_count = len(predictions)
        if is_distributed():
            device = predictions.device
            _sequence_count = torch.tensor(sequence_count).to(device)
            dist.all_reduce(_sequence_count, op=dist.ReduceOp.SUM)
            sequence_count = _sequence_count.item()
        self._total_sequence_count += sequence_count

    def _metric_mean(self, metric_sum):
        if self._total_sequence_count == 0:
            return 0.0
        return metric_sum / self._total_sequence_count

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        """
        # Parameters

        reset : `bool`, optional (default = `False`)
            Reset any accumulators or internal state.

        # Returns

        Dict[str, float]:
            A dictionary containing `ROUGE-1` .. `ROUGE-ngram_size` scores.
        """

        metrics = {}

        # ROUGE-N
        # Recall
        metrics.update(
            {
                f"ROUGE-{i}_R": self._metric_mean(self._total_rouge_n_recalls[i])
                for i in range(1, self._ngram_size + 1)
            }
        )

        # Precision
        metrics.update(
            {
                f"ROUGE-{i}_P": self._metric_mean(self._total_rouge_n_precisions[i])
                for i in range(1, self._ngram_size + 1)
            }
        )

        # F1
        metrics.update(
            {
                f"ROUGE-{i}_F1": self._metric_mean(self._total_rouge_n_f1s[i])
                for i in range(1, self._ngram_size + 1)
            }
        )

        # ROUGE-L
        # F1
        metrics["ROUGE-L"] = self._metric_mean(self._total_rouge_l_f1)

        if reset:
            self.reset()

        return metrics
