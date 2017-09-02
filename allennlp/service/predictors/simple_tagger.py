from typing import Any, Dict

from overrides import overrides
import numpy

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers import WordTokenizer
from allennlp.models import Model
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('simple-tagger')
class SimpleTaggerPredictor(Predictor):
    """
    Wrapper for the :class:`~allennlp.models.bidaf.SimpleTagger` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super(SimpleTaggerPredictor, self).__init__(model, dataset_reader)
        self._tokenizer = WordTokenizer()

    @overrides
    def _json_to_instance(self, json: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``
        and returns JSON that looks like
        ``{"tags": [...], "class_probabilities": [[...], ..., [...]]}``
        """
        sentence = json["sentence"]
        tokens, _ = self._tokenizer.tokenize(sentence)
        return self._dataset_reader.text_to_instance(tokens)

    @overrides
    def _predictions_to_json(self, model_outputs: Dict[str, Any]) -> JsonDict:
        predictions = model_outputs['class_probabilities']
        argmax_indices = numpy.argmax(predictions, axis=-1)
        tags = [self._model.vocab.get_token_from_index(x, namespace="labels")
                for x in argmax_indices]
        model_outputs['tags'] = tags
        return super(SimpleTaggerPredictor, self)._predictions_to_json(model_outputs)
