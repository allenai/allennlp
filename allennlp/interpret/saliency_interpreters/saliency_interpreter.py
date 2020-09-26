from allennlp.common import Registrable
from allennlp.common.util import JsonDict
from allennlp.predictors import Predictor


class SaliencyInterpreter(Registrable):
    """
    A `SaliencyInterpreter` interprets an AllenNLP Predictor's outputs by assigning a saliency
    score to each input token.
    """

    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor

    def saliency_interpret_from_json(self, inputs: JsonDict) -> JsonDict:
        """
        This function finds saliency values for each input token.

        # Parameters

        inputs : `JsonDict`
            The input you want to interpret (the same as the argument to a Predictor, e.g., predict_json()).

        # Returns

        interpretation : `JsonDict`
            Contains the normalized saliency values for each input token. The dict has entries for
            each instance in the inputs JsonDict, e.g., `{instance_1: ..., instance_2:, ... }`.
            Each one of those entries has entries for the saliency of the inputs, e.g.,
            `{grad_input_1: ..., grad_input_2: ... }`.
        """
        raise NotImplementedError("Implement this for saliency interpretations")
