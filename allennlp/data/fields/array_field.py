from typing import Dict

import numpy
import torch
from torch.autograd import Variable
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
    def as_tensor(self,
                  padding_lengths: Dict[str, int],
                  cuda_device: int = -1,
                  for_training: bool = True) -> torch.Tensor:
        max_shape = [padding_lengths["dimension_{}".format(i)]
                     for i in range(len(padding_lengths))]

        return_array = numpy.ones(max_shape, "float32") * self.padding_value

        # If the tensor has a different shape from the largest tensor, pad dimensions with zeros to
        # form the right shaped list of slices for insertion into the final tensor.
        slicing_shape = list(self.array.shape)
        if len(self.array.shape) < len(max_shape):
            slicing_shape = slicing_shape + [0 for _ in range(len(max_shape) - len(self.array.shape))]
        slices = [slice(0, x) for x in slicing_shape]
        return_array[slices] = self.array
        tensor = Variable(torch.from_numpy(return_array), volatile=not for_training)
        return tensor if cuda_device == -1 else tensor.cuda(cuda_device)

    @overrides
    def empty_field(self):  # pylint: disable=no-self-use
        # Pass the padding_value, so that any outer field, e.g., `ListField[ArrayField]` uses the
        # same padding_value in the padded ArrayFields
        return ArrayField(numpy.array([], dtype="float32"), padding_value=self.padding_value)
