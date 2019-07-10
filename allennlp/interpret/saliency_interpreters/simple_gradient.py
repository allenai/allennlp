# pylint: disable=protected-access
from typing import List
import math
import numpy
from allennlp.common.util import JsonDict, sanitize
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.interpret import SaliencyInterpreter

@SaliencyInterpreter.register('simple-gradients-interpreter')
class SimpleGradient(SaliencyInterpreter):
    def saliency_interpret_from_json(self, inputs: JsonDict) -> JsonDict:
        """
        Interprets the model's prediction for inputs.
        Gets the gradients of the loss with respect to the input
        and returns those gradients normalized and sanitized.
        """
        # Convert inputs to labeled instances
        labeled_instances = self.predictor.inputs_to_labeled_instances(inputs)

        # List of embedding inputs, used for multiplying gradient by the input for normalization
        embeddings_list = [] # type: List[np.ndarray]

        instances_with_grads = dict()
        for idx, instance in enumerate(labeled_instances):
            # Hook used for saving embeddings
            handle = self._register_forward_hook(embeddings_list)
            grads = self.predictor.get_gradients([instance])[0]
            handle.remove()

            # Gradients come back in the reverse order that they were sent into the network
            embeddings_list.reverse()
            for key, grad in grads.items():
                # Get number at the end of every gradient key
                # (they look like grad_input_[int], we're getting
                # this [int] part and subtracting 1 for zero-based indexing).
                # This is then used as an index into the reversed input array
                # to match up the gradient and its respective embedding.
                input_idx = int(key[-1]) - 1
                emb_grad = numpy.sum(grad * embeddings_list[input_idx], axis=1)
                norm = numpy.linalg.norm(emb_grad, ord=1)
                normalized_grad = [math.fabs(e) / norm for e in emb_grad]
                grads[key] = normalized_grad

            instances_with_grads['instance_' + str(idx + 1)] = grads
        return sanitize(instances_with_grads)

    def _register_forward_hook(self, embeddings_list: List):
        """ Finds all of the TextFieldEmbedders, and registers a forward hook
        onto them. When forward() is called, embeddings_list is filled with the
        embedding values. This is necessary because our normalization scheme
        multiplies the gradient by the embedding value.
        """

        def forward_hook(module, inp, output): # pylint: disable=unused-argument
            embeddings_list.append(output.squeeze(0).clone().detach().numpy())

        handle = None
        for module in self.predictor._model.modules():
            if isinstance(module, TextFieldEmbedder):
                handle = module.register_forward_hook(forward_hook)

        return handle
