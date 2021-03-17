"""
Code based almost entirely on
https://github.com/awaelchli/pytorch-lightning-snippets/commit/7db53f774715d635c59ef56f21a17634d246b2c5
"""

import torch
from torch import nn as nn
from typing import Tuple, List, Callable
from allennlp.sanity_checks.verification_base import VerificationBase
import logging

logger = logging.getLogger(__name__)


class NormalizationBiasVerification(VerificationBase):
    """
    Network layers with biases should not be combined with normalization layers,
    as the bias makes normalization ineffective and can lead to unstable training.
    This verification detects such combinations.
    """

    normalization_layers = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.GroupNorm,
        nn.LayerNorm,
    )

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self._hook_handles: List[Callable] = []
        self._module_sequence: List[Tuple[str, nn.Module]] = []
        self._detected_pairs: List[Tuple] = []

    @property
    def detected_pairs(self) -> List[Tuple]:
        return self._detected_pairs

    def check(self, inputs) -> bool:  # type: ignore
        inputs = self._get_inputs_copy(inputs)
        self.register_hooks()
        # trigger the hooks and collect sequence of layers
        self._model_forward(inputs)
        self.destroy_hooks()
        self.collect_detections()
        return not self._detected_pairs

    def collect_detections(self):
        detected_pairs = []
        for (name0, mod0), (name1, mod1) in zip(
            self._module_sequence[:-1], self._module_sequence[1:]
        ):
            bias = getattr(mod0, "bias", None)
            detected = (
                isinstance(mod1, self.normalization_layers)
                and mod1.training
                and isinstance(bias, torch.Tensor)
                and bias.requires_grad
            )
            if detected:
                detected_pairs.append((name0, name1))
        self._detected_pairs = detected_pairs
        if detected_pairs:
            logger.warning(self._verification_message())
        return detected_pairs

    def _verification_message(self):
        if self._detected_pairs:
            message = "\n\nThe model failed the NormalizationBiasVerification check:"
            for pair in self._detected_pairs:
                message += (
                    f"\n  * Detected a layer '{pair[0]}' with bias followed by"
                    f" a normalization layer '{pair[1]}'."
                )
            message += (
                "\n\nThis makes the normalization ineffective and can lead to unstable training. "
                "Either remove the normalization or turn off the bias.\n\n"
            )
        else:
            message = "\nThe model passed the NormalizationBiasVerification check."
        return message

    def register_hooks(self):
        hook_handles = []
        for name, module in self.model.named_modules():
            handle = module.register_forward_hook(self._create_hook(name))
            hook_handles.append(handle)
        self._hook_handles = hook_handles

    def _create_hook(self, module_name) -> Callable:
        def hook(module, inp_, out_):
            self._module_sequence.append((module_name, module))

        return hook

    def destroy_hooks(self):
        for hook in self._hook_handles:
            hook.remove()
        self._hook_handles = []
