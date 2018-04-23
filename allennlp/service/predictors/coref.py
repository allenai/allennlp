from typing import List, Tuple

from overrides import overrides

from allennlp.common.util import get_spacy_model, JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.service.predictors.predictor import Predictor


@Predictor.register("coreference-resolution")
class CorefPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.coreference_resolution.CoreferenceResolver` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

        # We have to use spacy to tokenise our document here, because we need
        # to also know sentence boundaries to propose valid mentions.
        self._spacy = get_spacy_model("en_core_web_sm", pos_tags=True, parse=True, ner=False)

    # pylint: disable=arguments-differ
    @overrides
    def predict(self, document: str, cuda_device: int = -1) -> JsonDict: # type: ignore
        """
        Predict the coreference clusters in the given document.

        Parameters
        ----------
        document : ``str``
            A string representation of a document.

        Returns
        -------
        A dictionary representation of the predicted coreference clusters.

        .. code-block:: js

            {
            "document": [tokenised document text]
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
        instance, results_dict = self._build_instance(document)
        outputs = self._model.forward_on_instance(instance, cuda_device)
        results_dict.update(outputs)
        return sanitize(results_dict)

    @overrides
    def predict_batch(self, inputs: List[JsonDict], cuda_device: int = -1):
        instances = [self._build_instance(**parameters) for parameters in inputs]
        return self._default_predict_batch(instances, cuda_device)

    def _build_instance(self, document: str) -> Tuple[Instance, JsonDict]:
        spacy_document = self._spacy(document)
        sentences = [[token.text for token in sentence] for sentence in spacy_document.sents]
        flattened_sentences = [word for sentence in sentences for word in sentence]

        results_dict: JsonDict = {"document": flattened_sentences}
        instance = self._dataset_reader.text_to_instance(sentences)
        return instance, results_dict
