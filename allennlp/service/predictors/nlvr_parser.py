from typing import Tuple
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('nlvr-parser')
class NlvrParserPredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        sentence = json_dict['sentence']
        worlds = json_dict['worlds']
        instance = self._dataset_reader.text_to_instance(sentence, worlds)
        return instance, {}
