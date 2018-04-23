from typing import Tuple
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor

@Predictor.register('simple_seq2seq')
class SimpleSeq2SeqPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.encoder_decoder.simple_seq2seq` model.
    """

    # pylint: disable=arguments-differ
    @overrides
    def predict(self, source: str, cuda_device: int = -1) -> JsonDict: # type: ignore
        return super().predict(source=source, cuda_device=cuda_device)

    # pylint: disable=arguments-differ
    @overrides
    def _build_instance(self, source: str) -> Tuple[Instance, JsonDict]: # type: ignore
        """
        Expects JSON that looks like ``{"source": "..."}``.
        """
        return self._dataset_reader.text_to_instance(source), {}
