from allennlp.data.fields import TextField
from allennlp.service.predictors import Predictor, JsonDict, sanitize


@Predictor.register('simple_tagger')
class SimpleTaggerPredictor(Predictor):
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        sentence = inputs["sentence"]

        tokens = TextField(self.tokenizer.tokenize(sentence),
                           token_indexers=self.token_indexers)

        return sanitize(self.model.tag(tokens))
