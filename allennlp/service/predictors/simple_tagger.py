from allennlp.common.util import JsonDict, sanitize
from allennlp.data.fields import TextField
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('simple-tagger')
class SimpleTaggerPredictor(Predictor):
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        sentence = inputs["sentence"]

        tokens = TextField(self.tokenizer.tokenize(sentence)[0],
                           token_indexers=self.token_indexers)

        return sanitize(self.model.tag(tokens))
