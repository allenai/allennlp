from typing import List 

import numpy

from allennlp.common.util import JsonDict, sanitize, normalize_by_total_score
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.interpret.saliency import SaliencyInterpreter

@SaliencyInterpreter.register('simple-gradients-interpreter')
class SimpleGradient(SaliencyInterpreter):
  def __init__(self, predictor):
    super().__init__(predictor)

  def interpret_from_json(self, inputs: JsonDict) -> JsonDict:
    """
    Gets the gradients of the loss with respect to the input and
    returns them normalized and sanitized.  
    """

    # Get labeled instances
    labeled_instances = self.predictor.inputs_to_labeled_instances(inputs)

    # *** list of embedding inputs *** 
    inputs = []

    instances_with_grads = dict()
    for idx, instance in enumerate(labeled_instances):

      # *** ADDED FOR EMBEDDING ***
      handle = self._register_forward_hook(inputs)

      grads = self.predictor.get_gradients([instance])[0]

      handle.remove()

      # *** reverse the inputs ***
      inputs.reverse()
      for key, grad in grads.items():

        # ****** L2 NORM ******
        # l2_grad = numpy.linalg.norm(grad, axis=1)
        # normalized_grad = normalize_by_total_score(l2_grad)

        # ****** EMBEDDING DOT PRODUCT ******

        # Here we get the number that is at the end of every gradient key
        # They look like grad_input_[int], we're getting this [int] part and
        # subtract 1 for zero based indexing. This is then used as an index into 
        # the reversed input array to match up the gradient and its respective
        # embedding. 
        input_idx = int(key[-1]) - 1

        assert grad.shape == inputs[input_idx].shape 

        emb_grad = numpy.sum(grad * inputs[input_idx], axis=1)

        assert emb_grad.shape[0] == grad.shape[0]

        normalized_grad = normalize_by_total_score(emb_grad)

        assert normalized_grad.shape[0] == grad.shape[0]
        assert numpy.absolute(numpy.sum(normalized_grad) - 1.0) < 1e-6
        # ****** END ******

        grads[key] = normalized_grad 

      instances_with_grads['instance_' + str(idx + 1)] = grads

    print("INSTANCES WITH GRADS")
    print("--------------------")
    print(instances_with_grads)
    return sanitize(instances_with_grads)

    # print('GRADS')
    # print('-----')
    # grads = self.predictor.get_gradients(labeled_instances)[0]
    # print(grads)

    # print('L2-NORM')
    # print('-------')
    # for key, grad in grads.items():
    #   l2_grad = numpy.linalg.norm(grad, axis=1)
    #   normalized_grad = normalize_by_total_score(l2_grad)
    #   grads[key] = normalized_grad
    # print(grads)

    # return sanitize(grads)

  def _register_forward_hook(self, inputs: List):
    """
    Register a forward hook on the embedding layer to get the embeddings.
    """
    def forward_hook(module, input, output):
      print("OUTPUT")
      print("------")
      print(output)

      inputs.append(output.squeeze(0).clone().detach().numpy())
    
    # Register the hook
    handle = None
    for module in self.predictor._model.modules():
        if isinstance(module, TextFieldEmbedder):
            handle = module.register_forward_hook(forward_hook)

    return handle 