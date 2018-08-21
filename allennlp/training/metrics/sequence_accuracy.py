from typing import Optional

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


@Metric.register("sequence_accuracy")
class SequenceAccuracy(Metric):
    """
    Sequence Top-K accuracy. Assumes integer labels, with
    each item to be classified having a single correct class.
    """
    def __init__(self) -> None:
        self.correct_count = 0.0
        self.total_count = 0.0

    # Note: See preprocess.py.
    def recall(beams, s2):
       retval = 0.
       for w in s2:
          stillsearch = True
          for s1 in beams:
             if stillsearch and (w in s1):
                retval += 1./float(len(s2))
                stillsearch = False
       return retval

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 end_index: int = -1):
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
        for i in range(batch_size):
            beams = predictions[i]
            cur_mask = mask[i]
            cur_gold = gold_labels[i]

            masked_gold = cur_gold * cur_mask
            #TODO(brendanr): Verify!
            cleaned_gold = [x for x in masked_gold if x != 0 and x != end_index]

            retval = 0.
            for w in cleaned_gold:
               stillsearch = True
               for beam in beams:
                  masked_beam = beam * cur_mask
                  s2 = [x for x in masked_beam if x != 0 and x != end_index]
                  if stillsearch and (w in beam):
                     retval += 1./float(len(s2))
                     stillsearch = False
            correct += retval

        self.correct_count += correct
        self.total_count += predictions.size()[0]

        return

        expanded_size = list(gold_labels.size())
        expanded_size.insert(1, k)
        expanded_gold = gold_labels.unsqueeze(1).expand(expanded_size)

        if mask is not None:
            expanded_mask = mask.unsqueeze(1).expand(expanded_size)
            masked_gold = expanded_mask * expanded_gold
            masked_predictions = expanded_mask * predictions
        else:
            masked_gold = expanded_gold
            masked_predictions = predictions

        eqs = masked_gold.eq(masked_predictions)
        matches_per_question = eqs.min(dim=2)[0]
        some_match = matches_per_question.max(dim=1)[0]
        correct = some_match.sum().item()

        self.total_count += predictions.size()[0]
        self.correct_count += correct

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        accuracy = float(self.correct_count) / float(self.total_count)
        if reset:
            self.reset()
        return accuracy

    @overrides
    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0
