from allennlp.common.util import JsonDict, sanitize
from allennlp.data.dataset_readers.squad import SquadReader
from allennlp.data.fields import TextField
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('machine-comprehension')
class BidafPredictor(Predictor):
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        question_text = inputs["question"]
        passage_text = inputs["passage"]

        question_tokens, _ = self.tokenizer.tokenize(question_text)
        passage_tokens, _ = self.tokenizer.tokenize(passage_text)
        passage_tokens.append(SquadReader.STOP_TOKEN)

        question = TextField(question_tokens, token_indexers=self.token_indexers)
        passage = TextField(passage_tokens, token_indexers=self.token_indexers)

        return sanitize(self.model.predict_span(question, passage))
