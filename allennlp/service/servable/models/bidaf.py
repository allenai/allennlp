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

        output_dict = self.model.predict_span(question, passage)

        # best_span is np.int64, we need to get pure python types so we can serialize them
        output_dict["best_span"] = [x.item() for x in output_dict["best_span"]]

        # similarly, the probability tensors must be converted to lists
        output_dict["span_start_probs"] = output_dict["span_start_probs"].tolist()
        output_dict["span_end_probs"] = output_dict["span_end_probs"].tolist()

        return output_dict
