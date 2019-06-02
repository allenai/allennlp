from allennlp.common.util import JsonDict, sanitize, normalize
from allennlp.interpretation import Interpreter

@Interpreter.register('simple-gradient-interpreter')
class SimpleGradient(Interpreter):
  def __init__(self, predictor):
    super().__init__(predictor)

  def interpret_from_json(self, inputs: JsonDict) -> JsonDict:
    """
    Gets the gradients of the loss with respect to the input and
    returns them normalized and sanitized.  
    """
    return sanitize(normalize(self.get_gradients(self.get_model_predictions(inputs))[0]))
