from allennlp.common.util import JsonDict, sanitize, normalize
from allennlp.common import Registrable 

class Interpreter(Registrable):
    def __init__(self, predictor):
        self.predictor = predictor 

    def interpret_from_json(self, inputs: JsonDict) -> JsonDict:
        """
        Gets the gradients of the loss with respect to the input and
        returns them normalized and sanitized.  
        """
        raise NotImplementedError("You should implement this if you want to do model interpretations")
