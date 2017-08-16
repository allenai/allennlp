from allennlp.data.fields import TextField
from allennlp.service.predictors import Predictor, JsonDict, sanitize

@Predictor.register('decomposable_attention')
class DecomposableAttentionPredictor(Predictor):
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        premise_text = inputs["premise"]
        hypothesis_text = inputs["hypothesis"]

        premise = TextField(self.tokenizer.tokenize(premise_text),
                            token_indexers=self.token_indexers)
        hypothesis = TextField(self.tokenizer.tokenize(hypothesis_text),
                               token_indexers=self.token_indexers)

        return sanitize(self.model.predict_entailment(premise, hypothesis))
