from typing import Dict

import numpy
from overrides import overrides

from allennlp.data.fields.field import Field


class ArrayField(Field[numpy.ndarray]):
    """
    A class representing an array, which could have arbitrary dimensions.
    A batch of these arrays are padded to the max dimension length in the batch
    for each dimension.
    """
    def __init__(self, array: numpy.ndarray, padding_value: int = 0) -> None:
        self.array = array
        self.padding_value = padding_value

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {"dimension_" + str(i): shape
                for i, shape in enumerate(self.array.shape)}

    @overrides
    def as_array(self, padding_lengths: Dict[str, int]) -> numpy.ndarray:
        max_shape = [padding_lengths["dimension_{}".format(i)]
                     for i in range(len(padding_lengths))]

        return_array = numpy.ones(max_shape, "float32") * self.padding_value

        # If the array has a different shape from the largest
        # array, pad dimensions with zeros to form the right
        # shaped list of slices for insertion into the final array.
        slicing_shape = list(self.array.shape)
        if len(self.array.shape) < len(max_shape):
            slicing_shape = slicing_shape + [0 for _ in range(len(max_shape) - len(self.array.shape))]
        slices = [slice(0, x) for x in slicing_shape]
        return_array[slices] = self.array
        return return_array

    @overrides
    def empty_field(self):  # pylint: disable=no-self-use
        return ArrayField(numpy.array([], dtype="float32"))
