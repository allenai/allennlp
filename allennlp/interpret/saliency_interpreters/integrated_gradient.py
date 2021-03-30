import math
from typing import List, Dict, Any

import numpy
import torch

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.interpret.saliency_interpreters.saliency_interpreter import SaliencyInterpreter
from allennlp.nn import util


@SaliencyInterpreter.register("integrated-gradient")
class IntegratedGradient(SaliencyInterpreter):
    """
    Interprets the prediction using Integrated Gradients (https://arxiv.org/abs/1703.01365)

    Registered as a `SaliencyInterpreter` with name "integrated-gradient".
    """

    def saliency_interpret_from_json(self, inputs: JsonDict) -> JsonDict:
        # Convert inputs to labeled instances
        labeled_instances = self.predictor.json_to_labeled_instances(inputs)

        instances_with_grads = dict()
        for idx, instance in enumerate(labeled_instances):
            # Run integrated gradients
            grads = self._integrate_gradients(instance)

            # Normalize results
            for key, grad in grads.items():
                # The [0] here is undo-ing the batching that happens in get_gradients.
                embedding_grad = numpy.sum(grad[0], axis=1)
                norm = numpy.linalg.norm(embedding_grad, ord=1)
                normalized_grad = [math.fabs(e) / norm for e in embedding_grad]
                grads[key] = normalized_grad

            instances_with_grads["instance_" + str(idx + 1)] = grads

        return sanitize(instances_with_grads)

    def _register_hooks(self, alpha: int, embeddings_list: List, token_offsets: List):
        """
        Register a forward hook on the embedding layer which scales the embeddings by alpha. Used
        for one term in the Integrated Gradients sum.

        We store the embedding output into the embeddings_list when alpha is zero.  This is used
        later to element-wise multiply the input by the averaged gradients.
        """

        def forward_hook(module, inputs, output):
            # Save the input for later use. Only do so on first call.
            if alpha == 0:
                embeddings_list.append(output.squeeze(0).clone().detach())

            # Scale the embedding by alpha
            output.mul_(alpha)

        def get_token_offsets(module, inputs, outputs):
            offsets = util.get_token_offsets_from_text_field_inputs(inputs)
            if offsets is not None:
                token_offsets.append(offsets)

        # Register the hooks
        handles = []
        embedding_layer = self.predictor.get_interpretable_layer()
        handles.append(embedding_layer.register_forward_hook(forward_hook))
        text_field_embedder = self.predictor.get_interpretable_text_field_embedder()
        handles.append(text_field_embedder.register_forward_hook(get_token_offsets))
        return handles

    def _integrate_gradients(self, instance: Instance) -> Dict[str, numpy.ndarray]:
        """
        Returns integrated gradients for the given [`Instance`](../../data/instance.md)
        """
        ig_grads: Dict[str, Any] = {}

        # List of Embedding inputs
        embeddings_list: List[torch.Tensor] = []
        token_offsets: List[torch.Tensor] = []

        # Use 10 terms in the summation approximation of the integral in integrated grad
        steps = 10

        # Exclude the endpoint because we do a left point integral approximation
        for alpha in numpy.linspace(0, 1.0, num=steps, endpoint=False):
            handles = []
            # Hook for modifying embedding value
            handles = self._register_hooks(alpha, embeddings_list, token_offsets)

            try:
                grads = self.predictor.get_gradients([instance])[0]
            finally:
                for handle in handles:
                    handle.remove()

            # Running sum of gradients
            if ig_grads == {}:
                ig_grads = grads
            else:
                for key in grads.keys():
                    ig_grads[key] += grads[key]

        # Average of each gradient term
        for key in ig_grads.keys():
            ig_grads[key] /= steps

        # Gradients come back in the reverse order that they were sent into the network
        embeddings_list.reverse()
        token_offsets.reverse()
        embeddings_list = self._aggregate_token_embeddings(embeddings_list, token_offsets)

        # Element-wise multiply average gradient by the input
        for idx, input_embedding in enumerate(embeddings_list):
            key = "grad_input_" + str(idx + 1)
            ig_grads[key] *= input_embedding

        return ig_grads
