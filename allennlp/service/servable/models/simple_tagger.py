from allennlp.data.fields import TextField
from allennlp.service.servable import Servable, JsonDict


@Servable.register('simple_tagger')
class SimpleTaggerServable(Servable):
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        sentence = inputs["sentence"]

        tokens = TextField(self.tokenizer.tokenize(sentence),
                           token_indexers=self.token_indexers)

        return self.model.tag(tokens)
