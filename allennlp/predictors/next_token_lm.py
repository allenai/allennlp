from typing import Dict

from copy import deepcopy
from overrides import overrides
import numpy

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.data.fields import LabelField
from allennlp.predictors.predictor import Predictor


@Predictor.register('next_token_lm')
class NextTokenLMPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence" : sentence})

    @overrides
    def predictions_to_labeled_instances(self,
                                         instance: Instance,
                                         outputs: Dict[str, numpy.ndarray]):
        new_instance = deepcopy(instance)
        new_instance.add_field('target_ids',
                               LabelField(outputs['words'][0]))
        return [new_instance]

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        """
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence=sentence)
