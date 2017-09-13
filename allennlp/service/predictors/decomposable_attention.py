from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('textual-entailment')
class DecomposableAttentionPredictor(Predictor):
    """
    Wrapper for the :class:`~allennlp.models.bidaf.DecomposableAttention` model.
    """
    @overrides
    def _json_to_instance(self, json: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"premise": "...", "hypothesis": "..."}``.
        """
        premise_text = json["premise"]
        hypothesis_text = json["hypothesis"]
        return self._dataset_reader.text_to_instance(premise_text, hypothesis_text)
