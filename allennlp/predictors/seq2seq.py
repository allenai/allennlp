from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register("seq2seq")
class Seq2SeqPredictor(Predictor):
    """
    Predictor for sequence to sequence models, including
    [`ComposedSeq2Seq`](../models/encoder_decoders/composed_seq2seq.md) and
    [`SimpleSeq2Seq`](../models/encoder_decoders/simple_seq2seq.md) and
    [`CopyNetSeq2Seq`](../models/encoder_decoders/copynet_seq2seq.md).
    """

    def predict(self, source: str) -> JsonDict:
        return self.predict_json({"source": source})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"source": "..."}`.
        """
        source = json_dict["source"]
        return self._dataset_reader.text_to_instance(source)
