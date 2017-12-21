from typing import Tuple

from overrides import overrides

from allennlp.common.util import get_spacy_model
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.service.predictors.predictor import Predictor


@Predictor.register("coreference-resolution")
class CorefPredictor(Predictor):
    """
    Wrapper for the :class:`~allennlp.models.coreference_resolution.CoreferenceResolver` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

        # We have to use spacy to tokenise our document here, because we need
        # to also know sentence boundaries to propose valid mentions.
        self._spacy = get_spacy_model("en_core_web_sm", pos_tags=True, parse=True, ner=False)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:

        """
        Expects JSON that looks like ``{"document": "string of document text"}``
        and returns JSON that looks like:

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
        document = json_dict["document"]
        spacy_document = self._spacy(document)
        sentences = [[token.text for token in sentence] for sentence in spacy_document.sents]
        flattened_sentences = [word for sentence in sentences for word in sentence]

        results_dict: JsonDict = {"document": flattened_sentences}
        instance = self._dataset_reader.text_to_instance(sentences)
        return instance, results_dict
