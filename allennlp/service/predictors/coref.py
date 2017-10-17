from overrides import overrides

import spacy

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.service.predictors.predictor import Predictor


@Predictor.register("coref")
class CorefPredictor(Predictor):
    """
    Wrapper for the :class:`~allennlp.models.coreference_resolution.CoreferenceResolver` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._spacy = spacy.load("en", ner=False, tagger=False, vectors=False)

    def _json_to_instance(self, json: JsonDict) -> Instance:
        # We're overriding `predict_json` directly, so we don't need this.  But I'd rather have a
        # useless stub here then make the base class throw a RuntimeError instead of a
        # NotImplementedError - the checking on the base class is worth it.
        raise RuntimeError("this should never be called")

    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = -1) -> JsonDict:
        """
        Expects JSON that looks like ``{"document": "..."}``
        and returns JSON that looks like:

        .. code-block:: js

            {
            "document": [...]
            "clusters":
              [
                [
                  [start_index, end_index],
                  [start_index, end_index]
                ],
                [
                  [start_index, end_index],
                  [start_index, end_index],
                  [start_index, end_index],
                ],
                ....
              ]
            }
        """
        document = inputs["document"]
        spacy_document = self._spacy(document)
        sentences = [[token.text for token in sentence] for sentence in spacy_document.sents]
        flattened_sentences = [word for sentence in sentences for word in sentence]

        results: JsonDict = {"document": flattened_sentences}
        instance = self._dataset_reader.text_to_instance(sentences)
        output = self._model.forward_on_instance(instance, cuda_device)

        results["clusters"] = output["clusters"][0]
        return sanitize(results)
