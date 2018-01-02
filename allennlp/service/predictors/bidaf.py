from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('machine-comprehension')
class BidafPredictor(Predictor):
    """
    Wrapper for the :class:`~allennlp.models.bidaf.BidirectionalAttentionFlow` model.
    """
    @overrides
    def _json_to_instance(self, obj: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"question": "...", "passage": "..."}``.
        """
        question_text = obj["question"]
        passage_text = obj["passage"]
        return self._dataset_reader.text_to_instance(question_text, passage_text)
