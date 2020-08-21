from typing import Dict, Any, Union, Optional

import torch
import numpy as np
from overrides import overrides

from allennlp.data.fields.field import Field


class ArrayField(Field[torch.Tensor]):
    """
    A class representing a tensor, which could have arbitrary dimensions.
    A batch of these tensors are padded to the max dimension length in the batch
    for each dimension.
    """

    __slots__ = ["tensor", "padding_value"]

    def __init__(
        self,
        tensor: Union[torch.Tensor, np.ndarray],
        padding_value: Any = 0.0,
        dtype: Optional[Union[np.dtype, torch.dtype]] = None,
    ) -> None:
        if dtype is not None:
            if isinstance(tensor, np.ndarray):
                tensor = tensor.astype(dtype)
            elif isinstance(tensor, torch.Tensor):
                tensor = tensor.to(dtype)
            else:
                raise ValueError("Did not recognize the type of `tensor`.")
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)

        self.tensor = tensor
        self.padding_value = padding_value

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {"dimension_" + str(i): shape for i, shape in enumerate(self.tensor.size())}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        pad = [
            padding
            for i, dimension_size in reversed(list(enumerate(self.tensor.size())))
            for padding in [0, padding_lengths["dimension_" + str(i)] - dimension_size]
        ]
        return torch.nn.functional.pad(self.tensor, pad, value=self.padding_value)

    @overrides
    def empty_field(self):
        # Pass the padding_value, so that any outer field, e.g., `ListField[ArrayField]` uses the
        # same padding_value in the padded ArrayFields
        return ArrayField(
            torch.tensor([], dtype=self.tensor.dtype), padding_value=self.padding_value
        )

    def __str__(self) -> str:
        return f"ArrayField with shape: {self.tensor.size()} and dtype: {self.tensor.dtype}."

    def __len__(self):
        return 1 if len(self.tensor.size()) <= 0 else self.tensor.size(0)

    def __eq__(self, other) -> bool:
        if isinstance(self, other.__class__):
            return (
                torch.equal(self.tensor, other.tensor) and self.padding_value == other.padding_value
            )
        return NotImplemented
