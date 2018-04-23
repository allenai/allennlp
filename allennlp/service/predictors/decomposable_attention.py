from typing import Dict, List, Tuple
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('textual-entailment')
class DecomposableAttentionPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.DecomposableAttention` model.
    """

    # pylint: disable=arguments-differ
    @overrides
    def predict(self, premise: str, hypothesis: str, cuda_device: int = -1) -> JsonDict: # type: ignore
        """
        Predicts whether the hypothesis is entailed by the premise text.

        Parameters
        ----------
        premise : ``str``
            A passage representing what is assumed to be true.

        hypothesis : ``str``
            A sentence that may be entailed by the premise.

        Returns
        -------
        A dictionary where the key "label_probs" determines the probabilities of each of
        [entailment, contradiction, neutral].
        """
        instance = self._build_instance(premise, hypothesis)
        outputs = self._model.forward_on_instance(instance, cuda_device)
        return sanitize(outputs)

    @overrides
    def predict_batch(self, inputs: List[JsonDict], cuda_device: int = -1):
        instances: List[Tuple[Instance, Dict]] =\
            [(self._build_instance(**parameters), {}) for parameters in inputs]
        return self._default_predict_batch(instances, cuda_device)

    def _build_instance(self, premise: str, hypothesis: str) -> Instance:
        return self._dataset_reader.text_to_instance(premise, hypothesis)
