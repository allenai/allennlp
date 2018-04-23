from typing import Tuple, Dict
from overrides import overrides

from allennlp.common.util import sanitize
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('textual-entailment')
class DecomposableAttentionPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.DecomposableAttention` model.
    """

    def predict(self, premise: str, hypothesis: str, cuda_device = -1) -> Dict:
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
        return super().predict(premise=premise, hypothesis=hypothesis)

    @overrides
    def _build_instance(self, premise: str, hypothesis: str) -> Tuple[Instance, Dict]:
        """
        Expects JSON that looks like ``{"premise": "...", "hypothesis": "..."}``.
        """
        return self._dataset_reader.text_to_instance(premise, hypothesis), {}
