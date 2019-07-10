from allennlp.common.util import JsonDict
from allennlp.predictors import Predictor
from allennlp.common import Registrable

class SaliencyInterpreter(Registrable):
    """
    a ``SaliencyInterpreter`` interprets an AllenNLP Predictor's
    outputs by assigning a saliency score to each input token.
    This score is then visualized, e.g., the AllenNLP demos.
    """
    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor

    def saliency_interpret_from_json(self, inputs: JsonDict) -> JsonDict:
        raise NotImplementedError("Implement this for saliency interpretations")
