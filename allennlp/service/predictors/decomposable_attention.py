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
    def _json_to_instance(self, obj: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"premise": "...", "hypothesis": "..."}``.
        """
        premise_text = obj["premise"]
        hypothesis_text = obj["hypothesis"]
        return self._dataset_reader.text_to_instance(premise_text, hypothesis_text)
