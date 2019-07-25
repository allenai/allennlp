from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('masked_lm_predictor')
class MaskedLanguageModelPredictor(Predictor):

    def predict(self, sentence_with_masks: str) -> JsonDict:
        return self.predict_json({"sentence" : sentence_with_masks})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        """
        tokens = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence=sentence)
