from allennlp.common.util import JsonDict, sanitize, normalize
from allennlp.interpretation import Interpreter

@Interpreter.register('simple-gradients-interpreter')
class SimpleGradient(Interpreter):
  def __init__(self, predictor):
    super().__init__(predictor)

  def interpret_from_json(self, inputs: JsonDict) -> JsonDict:
    """
    Gets the gradients of the loss with respect to the input and
    returns them normalized and sanitized.  
    """
    return sanitize(normalize(self.predictor.get_gradients(self.predictor.inputs_to_labeled_instances(inputs))[0]))
