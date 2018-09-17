from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register('event2mind')
class Event2MindPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.event2mind` model.
    """

    def predict(self, source: str) -> JsonDict:
        """
        Given a source string of some event, returns a JSON dictionary
        containing, for each target type, the top predicted sequences as
        indices, as tokens and the log probability of each.

        To be more precise, the dictionary will have the following entries:
          {target_type}_top_k_predictions: ``List[List[int]]``
          {target_type}_top_k_predicted_tokens: ``List[List[str]]``
          {target_type}_top_k_log_probabilities: ``List[float]``

        By default ``target_type`` can be xreact, oreact and xintent.
        """
        return self.predict_json({"source" : source})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"source": "..."}``.
        """
        source = json_dict["source"]
        return self._dataset_reader.text_to_instance(source)
