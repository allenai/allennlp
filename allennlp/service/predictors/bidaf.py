from typing import Tuple, Dict
from overrides import overrides

from allennlp.common.util import sanitize
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('machine-comprehension')
class BidafPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.BidirectionalAttentionFlow` model.
    """

    @overrides
    def predict(self, question: str, passage: str, cuda_device: int = -1) -> Dict:
        """
        Make a machine comprehension prediction on the supplied input.
        See https://rajpurkar.github.io/SQuAD-explorer/ for more information about the machine comprehension task.

        Parameters
        ----------
        question : ``str``
            A question about the content in the supplied paragraph.  The question must be answerable by a
            span in the paragraph.
        passage : ``str``
            A paragraph of information relevant to the question.

        Returns
        -------
        A dictionary that represents the prediction made by the system.  The answer string will be under the
        "best_span_str" key.
        """
        return super().predict(question=question, passage=passage, cuda_device=cuda_device)

    @overrides
    def _build_instance(self, question: str, passage: str) -> Tuple[Instance, Dict]:
        """
        Expects JSON that looks like ``{"question": "...", "passage": "..."}``.
        """
        return self._dataset_reader.text_to_instance(question, passage), {}
