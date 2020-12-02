from typing import List, Dict

from overrides import overrides
import numpy

from allennlp.common.file_utils import cached_path
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.data.dataset_readers.visual_entailment import VisualEntailmentReader
from allennlp.data.fields import LabelField
from allennlp.predictors.predictor import Predictor


@Predictor.register("vilbert_ve")
class VisualEntailmentPredictor(Predictor):
    def predict(self, image: str, hypothesis: str) -> JsonDict:
        image = cached_path(image)
        return self.predict_json({"image": image, "hypothesis": hypothesis})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        image = cached_path(json_dict["image"])
        hypothesis = json_dict["hypothesis"]
        if isinstance(self._dataset_reader, VisualEntailmentReader):
            return self._dataset_reader.text_to_instance(image, hypothesis, use_cache=False)
        else:
            raise ValueError(
                f"Dataset reader is of type f{self._dataset_reader.__class__.__name__}. "
                f"Expected {VisualEntailmentReader.__name__}."
            )

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        new_instance = instance.duplicate()
        label = numpy.argmax(outputs["probs"])
        new_instance.add_field("label", LabelField(int(label), skip_indexing=True))
        return [new_instance]
