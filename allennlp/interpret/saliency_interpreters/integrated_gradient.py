# pylint: disable=protected-access
import math
from typing import List, Dict, Any
import numpy
from allennlp.common.util import JsonDict, sanitize
from allennlp.interpret.saliency_interpreters.saliency_interpreter import SaliencyInterpreter
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data import Instance

@SaliencyInterpreter.register('integrated-gradients-interpreter')
class IntegratedGradient(SaliencyInterpreter):
    """
    Interprets the prediction using Integrated Gradients (https://arxiv.org/abs/1703.01365)
    """
    def saliency_interpret_from_json(self, inputs: JsonDict) -> JsonDict:
        # Convert inputs to labeled instances
        labeled_instances = self.predictor.inputs_to_labeled_instances(inputs)

        instances_with_grads = dict()
        for idx, instance in enumerate(labeled_instances):
            # Run integrated gradients
            grads = self.integrate_gradients(instance)

            # Normalize results
            for key, grad in grads.items():
                emb_grad = numpy.sum(grad, axis=1)
                norm = numpy.linalg.norm(emb_grad, ord=1)
                normalized_grad = [math.fabs(e) / norm for e in emb_grad]
                grads[key] = normalized_grad

            instances_with_grads['instance_' + str(idx + 1)] = grads

        return sanitize(instances_with_grads)

    def _register_forward_hook(self, alpha: int, embeddings_list: List):
        """
        Register a forward hook on the embedding layer which scales
        the embeddings by alpha. Used for one term in the Integrated Gradients sum.

        We store the embedding output into the embeddings_list when alpha is zero.
        This is used later to element-wise multiply the input by the averaged gradients.
        """
        def forward_hook(module, inp, output): # pylint: disable=unused-argument
            # Save the input for later use. Only do so on first call.
            if alpha == 0:
                embeddings_list.append(output.squeeze(0).clone().detach().numpy())

            # Scale the embedding by alpha
            output.mul_(alpha)

        # Register the hook
        handle = None
        for module in self.predictor._model.modules():
            if isinstance(module, TextFieldEmbedder):
                handle = module.register_forward_hook(forward_hook)

        return handle

    def integrate_gradients(self, instance: Instance) -> Dict[str, numpy.ndarray]:
        """
        Returns integrated gradients for the given :class:`~allennlp.data.instance.Instance`
        """
        ig_grads: Dict[str, Any] = {}

        # List of Embedding inputs
        embeddings_list: List[numpy.ndarray] = []

        # Use 10 terms in the summation approximation of the integral in integrated grad
        steps = 10

        # Exclude the endpoint because we do a left point integral approximation
        for alpha in numpy.linspace(0, 1.0, num=steps, endpoint=False):
            # Hook for modifying embedding value
            handle = self._register_forward_hook(alpha, embeddings_list)

            grads = self.predictor.get_gradients([instance])[0]
            handle.remove()

            # Running sum of gradients
            if ig_grads == {}:
                ig_grads = grads
            else:
                for key in grads.keys():
                    ig_grads[key] += grads[key]

        # Average of each gradient term
        for key in ig_grads.keys():
            ig_grads[key] *= 1/steps

        # Gradients come back in the reverse order that they were sent into the network
        embeddings_list.reverse()

        # Element-wise multiply average gradient by the input
        for idx, iput in enumerate(embeddings_list):
            key = "grad_input_" + str(idx + 1)
            ig_grads[key] *= iput

        return ig_grads
