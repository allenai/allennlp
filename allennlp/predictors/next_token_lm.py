from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('next_token_lm')
class NextTokenLMPredictor(Predictor):
    def predict(self, sentence: str, target: str) -> JsonDict:
        return self.predict_json({"sentence" : sentence, "target": target})

    def predictions_to_labeled_instances(self,
                                         instance: Instance,
                                         outputs: Dict[str, numpy.ndarray]):
        new_instance = deepcopy(instance)
        new_instance.add_field('target_ids', LabelField(int(outputs['top_indices'][0])), skip_indexing=True)
        return [new_instance]

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        """
        sentence = json_dict["sentence"]
        target = json_dict["target"]
        return self._dataset_reader.text_to_instance(sentence=sentence, target=target)
