from allennlp.data.dataset_readers.squad import SquadReader
from allennlp.data.fields import TextField
from allennlp.service.servable import Servable, JsonDict


@Servable.register('bidaf')
class BidafServable(Servable):
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        question_text = inputs["question"]
        passage_text = inputs["passage"]

        question = TextField(self.tokenizer.tokenize(question_text), token_indexers=self.token_indexers)
        passage = TextField(self.tokenizer.tokenize(passage_text) + [SquadReader.STOP_TOKEN],
                            token_indexers=self.token_indexers)

        return self.model.predict_span(question, passage)
