import torch
from allennlp.nn import Initializer
from typing import Dict


@Initializer.register('pretrained')
class PretrainedInitializer(Initializer):
    def __init__(self,
                 weights_file: str,
                 name_overrides: Dict[str, str] = {}) -> None:
        self.weights = torch.load(weights_file)
        self.name_overrides = name_overrides

    def __call__(self, tensor: torch.Tensor, name: str) -> None:
        # Select the new name if it's being overridden
        if name in self.name_overrides:
            name = self.name_overrides[name]
        # We specifically copy the weights into the tensor
        # to ensure the tensor sizes are identical
        tensor.data[:] = self.weights[name]

    @classmethod
    def from_params(cls, params):
        # The `from_params` method has to be defined, otherwise the
        # `Initializer` `from_params` method recursively calls itself
        weights_file = params.pop('weights_file')
        name_overrides = params.pop('name_overrides', {})
        return cls(weights_file=weights_file,
                   name_overrides=name_overrides)
