import math

from typing import List
import numpy
import torch

from allennlp.common.util import JsonDict, sanitize
from allennlp.interpret.saliency_interpreters.saliency_interpreter import SaliencyInterpreter
from allennlp.nn import util


@SaliencyInterpreter.register("simple-gradient")
class SimpleGradient(SaliencyInterpreter):
    """
    Registered as a `SaliencyInterpreter` with name "simple-gradient".
    """

    def saliency_interpret_from_json(self, inputs: JsonDict) -> JsonDict:
        """
        Interprets the model's prediction for inputs.  Gets the gradients of the loss with respect
        to the input and returns those gradients normalized and sanitized.
        """
        labeled_instances = self.predictor.json_to_labeled_instances(inputs)

        instances_with_grads = dict()
        for idx, instance in enumerate(labeled_instances):
            # List of embedding inputs, used for multiplying gradient by the input for normalization
            embeddings_list: List[torch.Tensor] = []
            token_offsets: List[torch.Tensor] = []

            # Hook used for saving embeddings
            handles = self._register_hooks(embeddings_list, token_offsets)
            try:
                grads = self.predictor.get_gradients([instance])[0]
            finally:
                for handle in handles:
                    handle.remove()

            # Gradients come back in the reverse order that they were sent into the network
            embeddings_list.reverse()
            token_offsets.reverse()
            embeddings_list = self._aggregate_token_embeddings(embeddings_list, token_offsets)

            for key, grad in grads.items():
                # Get number at the end of every gradient key (they look like grad_input_[int],
                # we're getting this [int] part and subtracting 1 for zero-based indexing).
                # This is then used as an index into the reversed input array to match up the
                # gradient and its respective embedding.
                input_idx = int(key[-1]) - 1
                # The [0] here is undo-ing the batching that happens in get_gradients.
                emb_grad = numpy.sum(grad[0] * embeddings_list[input_idx][0], axis=1)
                norm = numpy.linalg.norm(emb_grad, ord=1)
                normalized_grad = [math.fabs(e) / norm for e in emb_grad]
                grads[key] = normalized_grad

            instances_with_grads["instance_" + str(idx + 1)] = grads
        return sanitize(instances_with_grads)

    def _register_hooks(self, embeddings_list: List, token_offsets: List):
        """
        Finds all of the TextFieldEmbedders, and registers a forward hook onto them. When forward()
        is called, embeddings_list is filled with the embedding values. This is necessary because
        our normalization scheme multiplies the gradient by the embedding value.
        """

        def forward_hook(module, inputs, output):
            embeddings_list.append(output.squeeze(0).clone().detach())

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
