from typing import Tuple, Dict
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor

@Predictor.register('simple_seq2seq')
class SimpleSeq2SeqPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.encoder_decoder.simple_seq2seq` model.
    """

    def predict(self, source: str, cuda_device = -1) -> Dict:
        return super().predict(source=source, cuda_device=cuda_device)

    @overrides
    def _build_instance(self, source: str) -> Tuple[Instance, Dict]:
        """
        Expects JSON that looks like ``{"source": "..."}``.
        """
        return self._dataset_reader.text_to_instance(source), {}
