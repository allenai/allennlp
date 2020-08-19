from typing import Optional

import sys

from overrides import overrides
import torch
import torch.distributed as dist

from allennlp.common.util import is_distributed
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


@Metric.register("unigram_recall")
class UnigramRecall(Metric):
    """
    Unigram top-K recall. This does not take word order into account. Assumes
    integer labels, with each item to be classified having a single correct
    class.
    """

    def __init__(self) -> None:
        self.correct_count = 0.0
        self.total_count = 0.0

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        end_index: int = sys.maxsize,
    ):
        """
        # Parameters

        predictions : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, k, sequence_length).
        gold_labels : `torch.Tensor`, required.
            A tensor of integer class label of shape (batch_size, sequence_length).
        mask : `torch.BoolTensor`, optional (default = `None`).
            A masking tensor the same size as `gold_labels`.
        """
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)
        device = predictions.device

        # Some sanity checks.
        if gold_labels.dim() != predictions.dim() - 1:
            raise ConfigurationError(
                "gold_labels must have dimension == predictions.dim() - 1 but "
                "found tensor of shape: {}".format(gold_labels.size())
            )
        if mask is not None and mask.size() != gold_labels.size():
            raise ConfigurationError(
                "mask must have the same size as predictions but "
                "found tensor of shape: {}".format(mask.size())
            )

        batch_size = predictions.size()[0]
        correct = 0.0
        for i in range(batch_size):
            beams = predictions[i]
            cur_gold = gold_labels[i]

            if mask is not None:
                masked_gold = cur_gold * mask[i]
            else:
                masked_gold = cur_gold
            cleaned_gold = [x for x in masked_gold if x not in (0, end_index)]

            retval = 0.0
            for word in cleaned_gold:
                stillsearch = True
                for beam in beams:
                    # word is from cleaned gold which doesn't have 0 or
                    # end_index, so we don't need to explicitly remove those
                    # from beam.
                    if stillsearch and word in beam:
                        retval += 1 / len(cleaned_gold)
                        stillsearch = False
            correct += retval

        _correct_count = correct
        _total_count = predictions.size()[0]

        if is_distributed():
            correct_count = torch.tensor(_correct_count).to(device)
            total_count = torch.tensor(_total_count).to(device)
            dist.all_reduce(correct_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
            _correct_count = correct_count.item()
            _total_count = total_count.item()

        self.correct_count += _correct_count
        self.total_count += _total_count

    def get_metric(self, reset: bool = False):
        """
        # Returns

        The accumulated recall.
        """

        recall = self.correct_count / self.total_count if self.total_count > 0 else 0
        if reset:
            self.reset()
        return {"unigram_recall": recall}

    @overrides
    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0
