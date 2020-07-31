from typing import Optional, List, Union

from overrides import overrides
import torch
import torch.distributed as dist

from allennlp.training.metrics.metric import Metric


@Metric.register("attachment_scores")
class AttachmentScores(Metric):
    """
    Computes labeled and unlabeled attachment scores for a
    dependency parse, as well as sentence level exact match
    for both labeled and unlabeled trees. Note that the input
    to this metric is the sampled predictions, not the distribution
    itself.

    # Parameters

    ignore_classes : `List[int]`, optional (default = `None`)
        A list of label ids to ignore when computing metrics.
    """

    def __init__(self, ignore_classes: List[int] = None) -> None:
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._exact_labeled_correct = 0.0
        self._exact_unlabeled_correct = 0.0
        self._total_words = 0.0
        self._total_sentences = 0.0

        self._ignore_classes: List[int] = ignore_classes or []

    def __call__(  # type: ignore
        self,
        predicted_indices: torch.Tensor,
        predicted_labels: torch.Tensor,
        gold_indices: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        """
        # Parameters

        predicted_indices : `torch.Tensor`, required.
            A tensor of head index predictions of shape (batch_size, timesteps).
        predicted_labels : `torch.Tensor`, required.
            A tensor of arc label predictions of shape (batch_size, timesteps).
        gold_indices : `torch.Tensor`, required.
            A tensor of the same shape as `predicted_indices`.
        gold_labels : `torch.Tensor`, required.
            A tensor of the same shape as `predicted_labels`.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of the same shape as `predicted_indices`.
        """
        detached = self.detach_tensors(
            predicted_indices, predicted_labels, gold_indices, gold_labels, mask
        )
        predicted_indices, predicted_labels, gold_indices, gold_labels, mask = detached

        if mask is None:
            mask = torch.ones_like(predicted_indices).bool()

        predicted_indices = predicted_indices.long()
        predicted_labels = predicted_labels.long()
        gold_indices = gold_indices.long()
        gold_labels = gold_labels.long()

        # Multiply by a mask denoting locations of
        # gold labels which we should ignore.
        for label in self._ignore_classes:
            label_mask = gold_labels.eq(label)
            mask = mask & ~label_mask

        correct_indices = predicted_indices.eq(gold_indices).long() * mask
        unlabeled_exact_match = (correct_indices + ~mask).prod(dim=-1)
        correct_labels = predicted_labels.eq(gold_labels).long() * mask
        correct_labels_and_indices = correct_indices * correct_labels
        labeled_exact_match = (correct_labels_and_indices + ~mask).prod(dim=-1)

        self._unlabeled_correct += correct_indices.sum()
        self._exact_unlabeled_correct += unlabeled_exact_match.sum()
        self._labeled_correct += correct_labels_and_indices.sum()
        self._exact_labeled_correct += labeled_exact_match.sum()
        self._total_sentences += correct_indices.size(0)
        self._total_words += correct_indices.numel() - (~mask).sum()

    def get_metric(
        self,
        reset: bool = False,
        world_size: int = 1,
        cuda_device: Union[int, torch.device] = torch.device("cpu"),
    ):
        """
        # Returns

        The accumulated metrics as a dictionary.
        """
        unlabeled_attachment_score = 0.0
        labeled_attachment_score = 0.0
        unlabeled_exact_match = 0.0
        labeled_exact_match = 0.0

        if world_size > 1:
            self._unlabeled_correct = torch.tensor(self._unlabeled_correct).to(cuda_device)
            self._exact_unlabeled_correct = torch.tensor(self._exact_unlabeled_correct).to(
                cuda_device
            )
            self._labeled_correct = torch.tensor(self._labeled_correct).to(cuda_device)
            self._exact_labeled_correct = torch.tensor(self._exact_labeled_correct).to(cuda_device)
            self._total_sentences = torch.tensor(self._total_sentences).to(cuda_device)
            self._total_words = torch.tensor(self._total_words).to(cuda_device)

            dist.all_reduce(self._unlabeled_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(self._exact_unlabeled_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(self._labeled_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(self._exact_labeled_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(self._total_sentences, op=dist.ReduceOp.SUM)
            dist.all_reduce(self._total_words, op=dist.ReduceOp.SUM)

        if self._total_words > 0.0:
            unlabeled_attachment_score = float(self._unlabeled_correct) / float(self._total_words)
            labeled_attachment_score = float(self._labeled_correct) / float(self._total_words)
        if self._total_sentences > 0:
            unlabeled_exact_match = float(self._exact_unlabeled_correct) / float(
                self._total_sentences
            )
            labeled_exact_match = float(self._exact_labeled_correct) / float(self._total_sentences)
        if reset:
            self.reset()
        metrics = {
            "UAS": unlabeled_attachment_score,
            "LAS": labeled_attachment_score,
            "UEM": unlabeled_exact_match,
            "LEM": labeled_exact_match,
        }
        # if world_size > 1:
        #    metrics = self._aggregate_metrics(metrics)
        return metrics

    @overrides
    def reset(self):
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._exact_labeled_correct = 0.0
        self._exact_unlabeled_correct = 0.0
        self._total_words = 0.0
        self._total_sentences = 0.0
