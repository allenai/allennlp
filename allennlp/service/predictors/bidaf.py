from typing import Dict, List, Tuple
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('machine-comprehension')
class BidafPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.BidirectionalAttentionFlow` model.
    """

    # pylint: disable=arguments-differ
    @overrides
    def predict(self, question: str, passage: str, cuda_device: int = -1) -> JsonDict: # type: ignore
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
        instance = self._build_instance(question, passage)
        outputs = self._model.forward_on_instance(instance, cuda_device)
        return sanitize(outputs)

    @overrides
    def predict_batch(self, inputs: List[JsonDict], cuda_device: int = -1):
        instances: List[Tuple[Instance, Dict]] =\
            [(self._build_instance(**parameters), {}) for parameters in inputs]
        return self._default_predict_batch(instances, cuda_device)

    def _build_instance(self, question: str, passage: str) -> Instance:
        return self._dataset_reader.text_to_instance(question, passage)
