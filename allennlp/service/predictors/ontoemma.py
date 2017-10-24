from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('ontoemma')
class OntoEmmaPredictor(Predictor):
    """
    Wrapper for the :class:`~allennlp.models.ontoemma` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super(OntoEmmaPredictor, self).__init__(model, dataset_reader)

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        outputs = self._model.forward_on_instance(instance)
        print(outputs)
        return sanitize(outputs)

    @overrides
    def _json_to_instance(self, json: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``
        and returns JSON that looks like
        ``{"tags": [...], "class_probabilities": [[...], ..., [...]]}``
        """
        input()
        source_ent_name = json["source_ent"]['canonical_name'].lower()
        target_ent_name = json["target_ent"]['canonical_name'].lower()
        label = json["label"]
        print("%i: \"%s\" and \"%s\"" % (label, source_ent_name, target_ent_name))
        return self._dataset_reader.text_to_instance(source_ent_name, target_ent_name, label)