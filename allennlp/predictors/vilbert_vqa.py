from typing import List, Dict

from overrides import overrides
import numpy

from allennlp.common.file_utils import cached_path
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register("vilbert_vqa")
class VilbertVqaPredictor(Predictor):
    def predict(self, image: str, sentence: str) -> JsonDict:
        image = cached_path(image)
        return self.predict_json({"question": sentence, "image": image})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        question = json_dict["question"]
        image = cached_path(json_dict["image"])
        return self._dataset_reader.text_to_instance(question, image, use_cache=False)

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        return [instance]   # TODO
