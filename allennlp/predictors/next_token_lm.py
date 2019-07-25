from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('next_token_lm')
class NextTokenLMPredictor(Predictor):

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence" : sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        """
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence=sentence)
