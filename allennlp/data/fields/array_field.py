from typing import Dict

import numpy
import torch
from overrides import overrides

from allennlp.data.fields.field import Field


class ArrayField(Field[numpy.ndarray]):
    """
    A class representing an array, which could have arbitrary dimensions.
    A batch of these arrays are padded to the max dimension length in the batch
    for each dimension.
    """

    def __init__(
        self, array: numpy.ndarray, padding_value: int = 0, dtype: numpy.dtype = numpy.float32
    ) -> None:
        self.array = array
        self.padding_value = padding_value
        self.dtype = dtype

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {"dimension_" + str(i): shape for i, shape in enumerate(self.array.shape)}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        max_shape = [padding_lengths["dimension_{}".format(i)] for i in range(len(padding_lengths))]

        # Convert explicitly to an ndarray just in case it's an scalar
        # (it'd end up not being an ndarray otherwise).
        # Also, the explicit dtype declaration for `asarray` is necessary for scalars.
        return_array = numpy.asarray(
            numpy.ones(max_shape, dtype=self.dtype) * self.padding_value, dtype=self.dtype
        )

        # If the tensor has a different shape from the largest tensor, pad dimensions with zeros to
        # form the right shaped list of slices for insertion into the final tensor.
        slicing_shape = list(self.array.shape)
        if len(self.array.shape) < len(max_shape):
            slicing_shape = slicing_shape + [
                0 for _ in range(len(max_shape) - len(self.array.shape))
            ]
        slices = tuple([slice(0, x) for x in slicing_shape])
        return_array[slices] = self.array
        tensor = torch.from_numpy(return_array)
        return tensor

    @overrides
    def empty_field(self):
        # Pass the padding_value, so that any outer field, e.g., `ListField[ArrayField]` uses the
        # same padding_value in the padded ArrayFields
        return ArrayField(
            numpy.array([], dtype=self.dtype), padding_value=self.padding_value, dtype=self.dtype
        )

    def __str__(self) -> str:
        return f"ArrayField with shape: {self.array.shape} and dtype: {self.dtype}."

    def __len__(self):
        return 1 if self.array.ndim == 0 else self.array.shape[0]
