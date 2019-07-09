import math
from typing import Dict
import torch
import numpy
from allennlp.common.util import JsonDict, sanitize
from allennlp.interpret.saliency_interpreters import SaliencyInterpreter
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.predictors import Predictor
from allennlp.data import Instance

@SaliencyInterpreter.register('smooth-gradient-interpreter')
class SmoothGradient(SaliencyInterpreter):
    """
    Interprets the prediction using SmoothGrad (https://arxiv.org/abs/1706.03825)
    """
    def __init__(self, predictor: Predictor):
        super().__init__(predictor)

    def saliency_interpret_from_json(self, inputs: JsonDict) -> JsonDict:
        # Convert inputs to labeled instances
        labeled_instances = self.predictor.inputs_to_labeled_instances(inputs)

        instances_with_grads = dict()
        for idx, instance in enumerate(labeled_instances):
            # Run smoothgrad
            grads = self.smooth_grads(instance)

            # Normalize results
            for key, grad in grads.items():
                # TODO (@Eric-Wallace), SmoothGrad is not using times input normalization.
                # Fine for now, but should fix for consistency.
                emb_grad = numpy.sum(grad, axis=1)
                norm = numpy.linalg.norm(emb_grad, ord=1)
                normalized_grad = [math.fabs(e) / norm for e in emb_grad]
                grads[key] = normalized_grad

            instances_with_grads['instance_' + str(idx + 1)] = grads

        return sanitize(instances_with_grads)

    def _register_forward_hook(self, stdev: int):
        """
        Register a forward hook on the embedding layer which adds random noise to every embedding.
        Used for one term in the SmoothGrad sum.
        """
        def forward_hook(inp, grad_in, grad_out): # pylint: disable=unused-argument
            # Random noise = N(0, stdev * (max-min))
            noise = torch.randn(output.shape).to(output.device) \
                * (stdev * (output.detach().max() - output.detach().min()))

            # Add the random noise
            output.add_(noise)

        # Register the hook
        handle = None
        for module in self.predictor._model.modules():
            if isinstance(module, TextFieldEmbedder):
                handle = module.register_forward_hook(forward_hook)

        return handle

    def smooth_grads(self, instance: Instance) -> Dict[str, numpy.ndarray]:
        # Hyperparameters
        stdev = 0.01
        num_samples = 25

        total_gradients = None
        for _ in range(num_samples):
            handle = self._register_forward_hook(stdev)
            grads = self.predictor.get_gradients([instance])[0]
            handle.remove()

            # Sum gradients
            if total_gradients is None:
                total_gradients = grads
            else:
                for key in grads.keys():
                    total_gradients[key] += grads[key]

        # Average the gradients
        for key in total_gradients.keys():
            total_gradients[key] /= num_samples

        return total_gradients
