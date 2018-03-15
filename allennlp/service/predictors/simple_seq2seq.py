from typing import Tuple
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor

@Predictor.register('simple_seq2seq')
class SimpleSeq2SeqPredictor(Predictor):
    """
    Wrapper for the :class:`~allennlp.models.encoder_decoder.simple_seq2seq` model.
    """
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        """
        Expects JSON that looks like ``{"source_string": "..."}``.
        """
        source_string = json_dict["source_string"]
        return self._dataset_reader.text_to_instance(source_string), {}
