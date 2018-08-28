from typing import Optional

from overrides import overrides
import sys
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


@Metric.register("unigram_recall")
class UnigramRecall(Metric):
    """
    Unigram top-K recall. Assumes integer labels, with
    each item to be classified having a single correct class.
    """
    def __init__(self) -> None:
        self.correct_count = 0.0
        self.total_count = 0.0

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 end_index: int = sys.maxsize):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, k, sequence_length).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, sequence_length).
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        # Some sanity checks.
        if gold_labels.dim() != predictions.dim() - 1:
            raise ConfigurationError("gold_labels must have dimension == predictions.dim() - 1 but "
                                     "found tensor of shape: {}".format(gold_labels.size()))
        if mask is not None and mask.size() != gold_labels.size():
            raise ConfigurationError("mask must have the same size as predictions but "
                                     "found tensor of shape: {}".format(mask.size()))

        k = predictions.size()[1]
        batch_size = predictions.size()[0]
        correct = 0.0
        # Note: See preprocess.py.
        for i in range(batch_size):
            beams = predictions[i]
            cur_gold = gold_labels[i]

            if mask is not None:
                masked_gold = cur_gold * mask[i]
            else:
                masked_gold = cur_gold
            #TODO(brendanr): Verify! Is 0 a valid index?
            cleaned_gold = [x for x in masked_gold if x != 0 and x != end_index]

            retval = 0.
            for w in cleaned_gold:
                stillsearch = True
                for beam in beams:
                    if mask is not None:
                        masked_beam = beam * mask[i]
                    else:
                        masked_beam = beam
                    # w is from cleaned gold which doesn't have 0 or end_index,
                    # so we don't need to explicitly remove those from
                    # masked_beam.
                    if stillsearch and (w in masked_beam):
                        retval += 1./float(len(cleaned_gold))
                        stillsearch = False
            correct += retval

        self.correct_count += correct
        self.total_count += predictions.size()[0]

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated recall.
        """
        recall = float(self.correct_count) / float(self.total_count)
        if reset:
            self.reset()
        return recall

    @overrides
    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0
