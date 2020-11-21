from typing import List, Dict

from overrides import overrides
import numpy

from allennlp.common.file_utils import cached_path
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.data.dataset_readers.vqav2 import VQAv2Reader
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
        if isinstance(self._dataset_reader, VQAv2Reader):
            return self._dataset_reader.text_to_instance(question, image, use_cache=False)
        else:
            raise ValueError(
                f"Dataset reader is of type f{self._dataset_reader.__class__.__name__}. "
                f"Expected {VQAv2Reader.__name__}."
            )

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        return [instance]  # TODO
