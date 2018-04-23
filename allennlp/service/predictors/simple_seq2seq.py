from typing import Dict, List, Tuple
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
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
        instance = self._build_instance(source)
        outputs = self._model.forward_on_instance(instance, cuda_device)
        return sanitize(outputs)

    @overrides
    def predict_batch(self, inputs: List[JsonDict], cuda_device: int = -1):
        instances: List[Tuple[Instance, Dict]] =\
            [(self._build_instance(**parameters), {}) for parameters in inputs]
        return self._default_predict_batch(instances, cuda_device)

    def _build_instance(self, source: str) -> Instance:
        return self._dataset_reader.text_to_instance(source)
