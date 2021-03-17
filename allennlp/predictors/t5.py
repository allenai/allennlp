from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register("t5")
class T5Predictor(Predictor):
    def predict(self, source: str) -> JsonDict:
        return self.predict_json({"source": source})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"source": "..."}`.
        """
        source = json_dict["source"]
        return self._dataset_reader.text_to_instance(source)

    @overrides
    def _finalize_output(self, output: JsonDict) -> JsonDict:
        predictions = output["predictions"]
        output["predicted_text"] = self._dataset_reader.tokenizer.tokenizer.batch_decode(  # type: ignore
            predictions, skip_special_tokens=True
        )
        return output

    @classmethod
    def from_pretrained(cls, model_name: str) -> "T5Predictor":
        from allennlp.data import Vocabulary
        from allennlp.data.dataset_readers import T5DatasetReader
        from allennlp.models import T5ForConditionalGeneration
        from allennlp.modules.transformer import T5

        vocab = Vocabulary.from_pretrained_transformer(model_name)
        reader = T5DatasetReader(model_name)
        model = T5ForConditionalGeneration(vocab, T5.from_pretrained_module(model_name))

        return cls(model, reader)
