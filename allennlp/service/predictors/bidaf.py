from allennlp.common.util import JsonDict, sanitize
from allennlp.data.fields import TextField
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('machine-comprehension')
class BidafPredictor(Predictor):
    """
    Wrapper for the :class:`~allennlp.models.bidaf.BidirectionalAttentionFlow` model.
    """
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        """
        Expects JSON that looks like ``{"question": "...", "passage": "..."}``
        and returns JSON that looks like
        ``{"best_span": "...", "best_span_str": "...", "span_start_probs": "...", "span_end_probs": "..."}``
        """
        question_text = inputs["question"]
        passage_text = inputs["passage"]

        question_tokens, _ = self.tokenizer.tokenize(question_text)
        passage_tokens, _ = self.tokenizer.tokenize(passage_text)

        question = TextField(question_tokens, token_indexers=self.token_indexers)
        passage = TextField(passage_tokens, token_indexers=self.token_indexers)

        prediction = self.model.predict_span(question, passage)

        # Add the question tokens, so the token intervals are useful
        prediction["tokens"] = passage_tokens

        return sanitize(prediction)
