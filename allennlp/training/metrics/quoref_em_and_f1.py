from typing import List, Union

from overrides import overrides

from allennlp.training.metrics.metric import Metric
from allennlp.training.metrics.drop_em_and_f1 import DropEmAndF1


@Metric.register("quoref")
class QuorefEmAndF1(DropEmAndF1):
    """
    The evaluation metric for Quoref is exactly the same as the one used for spans by the DROP metric. So we just
    transform the ground truth information into the appropriate format and call the DROP metric on it.
    """
    @overrides
    def __call__(self, prediction: Union[str, List], ground_truths: List[str]):  # type: ignore
        """
        Parameters
        ----------
        prediction: ``Union[str, List]``
            The predicted answer from the model evaluated.
            This could be a string, or a list of string
            when multiple spans are predicted as answer.
        ground_truths: ``List[str]``
            All the ground truth answers.
        """
        # Convert ground truths to DROP input format
        ground_truths = [{"spans": ground_truths}]
        super(QuorefEmAndF1, self).__call__(prediction, ground_truths)

    def __str__(self):
        return f"QuorefEmAndF1(em={self._total_em}, f1={self._total_f1})"
