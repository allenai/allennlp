from typing import Tuple

from overrides import overrides
from allennlp.data.dataset_readers.snli import SnliReader
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('textual-entailment')
class DecomposableAttentionPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.DecomposableAttention` model.
    """

    def predict(self, premise: str, hypothesis: str) -> JsonDict:
        """
        Predicts whether the hypothesis is entailed by the premise text.

        Parameters
        ----------
        premise : ``str``
            A passage representing what is assumed to be true.

        hypothesis : ``str``
            A sentence that may be entailed by the premise.

        Returns
        -------
        A dictionary where the key "label_probs" determines the probabilities of each of
        [entailment, contradiction, neutral].
        """
        return self.predict_json({"premise" : premise, "hypothesis": hypothesis})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        """
        Expects JSON that looks like ``{"premise": "...", "hypothesis": "..."}``.
        """
        premise_text = json_dict["premise"]
        hypothesis_text = json_dict["hypothesis"]
        snli_reader: SnliReader = self._dataset_reader   # type: ignore
        tokenizer = snli_reader._tokenizer # pylint: disable=protected-access

        return self._dataset_reader.text_to_instance(premise_text, hypothesis_text), {
                'premise_tokens': [token.text for token in tokenizer.tokenize(premise_text)],
                'hypothesis_tokens': [token.text for token in tokenizer.tokenize(hypothesis_text)]
        }
