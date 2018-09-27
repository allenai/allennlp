from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register('atis-parser')
class AtisParserPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.semantic_parsing.atis.AtisSemanticParser` model.
    """
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"utterance": "..."}``.
        """
        utterance = json_dict["utterance"]
        return self._dataset_reader.text_to_instance([utterance])
