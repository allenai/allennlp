from allennlp.common.util import JsonDict, sanitize
from allennlp.data.fields import TextField
from allennlp.service.predictors.predictor import Predictor

@Predictor.register('textual-entailment')
class DecomposableAttentionPredictor(Predictor):
    """
    Wrapper for the :class:`~allennlp.models.bidaf.DecomposableAttention` model.
    """
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        """
        Expects JSON that looks like ``{"premise": "...", "hypothesis": "..."}``
        and returns JSON that looks like
        ``{"label_probs": [entailment_prob, contradiction_prob, neutral_prob]}``
        """
        premise_text = inputs["premise"]
        hypothesis_text = inputs["hypothesis"]

        premise = TextField(self.tokenizer.tokenize(premise_text)[0],
                            token_indexers=self.token_indexers)
        hypothesis = TextField(self.tokenizer.tokenize(hypothesis_text)[0],
                               token_indexers=self.token_indexers)

        return sanitize(self.model.predict_entailment(premise, hypothesis))
