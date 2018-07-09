from typing import Tuple
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('esim')
class ESIMPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.esim.ESIM` model.
    """

    def predict(self, sentence1: str, sentence2: str) -> JsonDict:
        """
        Predicts whether the sentence2 is entailed by the sentence1 text.

        Parameters
        ----------
        sentence1 : ``str``
            A passage representing what is assumed to be true.

        sentence2 : ``str``
            A sentence that may be entailed by the sentence1.

        Returns
        -------
        A dictionary where the key "label_probs" determines the probabilities of each of
        [entailment, contradiction, neutral].
        """
        return self.predict_json({"sentence1" : sentence1, "sentence2": sentence2})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence1": "...", "sentence2": "..."}``.
        """
        sentence1_text = json_dict["sentence1"]
        sentence2_text = json_dict["sentence2"]
        return self._dataset_reader.text_to_instance(sentence1_text, sentence2_text)
