from typing import List

from overrides import overrides
from spacy.tokens import Doc

from allennlp.common.util import JsonDict
from allennlp.common.util import get_spacy_model
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


@Predictor.register("coreference-resolution")
class CorefPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.coreference_resolution.CoreferenceResolver` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)

        # We have to use spacy to tokenise our document here, because we need
        # to also know sentence boundaries to propose valid mentions.
        self._spacy = get_spacy_model(language, pos_tags=True, parse=True, ner=False)

    def predict(self, document: str) -> JsonDict:
        """
        Predict the coreference clusters in the given document.

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

        Parameters
        ----------
        document : ``str``
            A string representation of a document.

        Returns
        -------
        A dictionary representation of the predicted coreference clusters.
        """
        return self.predict_json({"document" : document})

    def predict_tokenized(self, tokenized_document: List[str]) -> JsonDict:
        """
        Predict the coreference clusters in the given document.

        Parameters
        ----------
        tokenized_document : ``List[str]``
            A list of words representation of a tokenized document.

        Returns
        -------
        A dictionary representation of the predicted coreference clusters.
        """
        instance = self._words_list_to_instance(tokenized_document)
        return self.predict_instance(instance)

    def _words_list_to_instance(self, words: List[str]) -> Instance:
        """
        Create an instance from words list represent an already tokenized document,
        for skipping tokenization when that information already exist for the user
        """
        spacy_document = Doc(self._spacy.vocab, words=words)
        for pipe in filter(None, self._spacy.pipeline):
            pipe[1](spacy_document)

        sentences = [[token.text for token in sentence] for sentence in spacy_document.sents]  # pylint: disable=not-an-iterable
        instance = self._dataset_reader.text_to_instance(sentences)
        return instance

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"document": "string of document text"}``
        """
        document = json_dict["document"]
        spacy_document = self._spacy(document)
        sentences = [[token.text for token in sentence] for sentence in spacy_document.sents]
        instance = self._dataset_reader.text_to_instance(sentences)
        return instance
