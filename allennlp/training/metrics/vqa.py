from typing import Union

import torch
from overrides import overrides

from allennlp.training.metrics.metric import Metric
import torch.distributed as dist


@Metric.register("vqa")
class VqaMeasure(Metric):
    """Compute the VQA metric, as described in
    https://www.semanticscholar.org/paper/VQA%3A-Visual-Question-Answering-Agrawal-Lu/97ad70a9fa3f99adf18030e5e38ebe3d90daa2db

    In VQA, we take the answer with the highest score, and then we find out how often
    humans decided this was the right answer. The accuracy score for an answer is
    `min(1.0, human_count / 3)`.

    This metric takes the logits from the models, i.e., a score for each possible answer,
    and the labels for the question, together with their weights.
    """

    def __init__(self) -> None:
        self._sum_of_scores: Union[None, torch.Tensor] = None
        self._score_count: Union[None, torch.Tensor] = None

    @overrides
    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, label_weights: torch.Tensor):
        """
        # Parameters

        logits : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, num_classes).
        labels : `torch.Tensor`, required.
            A tensor of integer class label of shape (batch_size, num_labels).
        label_weights : `torch.Tensor`, required.
            A tensor of floats of shape (batch_size, num_labels), giving a weight or score to
            every one of the labels.
        """

        device = logits.device

        if self._sum_of_scores is None:
            self._sum_of_scores = torch.zeros([], device=device, dtype=label_weights.dtype)
        if self._score_count is None:
            self._score_count = torch.zeros([], device=device, dtype=torch.int32)

        logits, labels, label_weights = self.detach_tensors(logits, labels, label_weights)
        predictions = logits.argmax(dim=1)

        # Sum over dimension 1 gives the score per question. We care about the overall sum though,
        # so we sum over all dimensions.
        self._sum_of_scores += (label_weights * (labels == predictions.unsqueeze(-1))).sum()
        self._score_count += labels.size(0)

        from allennlp.common.util import is_distributed

        if is_distributed():
            dist.all_reduce(self._sum_of_scores, op=dist.ReduceOp.SUM)
            dist.all_reduce(self._score_count, op=dist.ReduceOp.SUM)

    @overrides
    def get_metric(self, reset: bool = False):
        """
        # Returns

        score : `float`
        """
        from allennlp.common.util import nan_safe_tensor_divide

        return {"score": nan_safe_tensor_divide(self._sum_of_scores, self._score_count).item()}

    @overrides
    def reset(self) -> None:
        self._sum_of_scores = None
        self._score_count = None
