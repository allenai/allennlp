from typing import Dict

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.data.fields.field import Field

class TensorField(Field[torch.Tensor]):
    """
    A ``TensorField`` is a PyTorch tensor of any type, and can be used to integrate
    arbitrary features that you might have with your data.
    """
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        if not isinstance(tensor, torch.Tensor):
            raise ConfigurationError("TensorFields must be passed torch.Tensors. "
                                     "Found: {} with type {}.".format(tensor, type(tensor)))

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:  # pylint: disable=no-self-use
        return {}

    @overrides
    def as_tensor(self,
                  padding_lengths: Dict[str, int],
                  cuda_device: int = -1) -> torch.Tensor:
        # pylint: disable=unused-argument,not-callable
        return self.tensor if cuda_device == -1 else self.tensor.cuda(cuda_device)

    @overrides
    def empty_field(self):
        return TensorField(self.tensor.new([]))

    def __str__(self) -> str:
        return f"TensorField storing: {self.tensor}."
