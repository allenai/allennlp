"""
A wrapper that unrolls the second (time) dimension of a tensor
into the first (batch) dimension, applies some other ``Module``,
and then rolls the time dimension back up.
"""
import torch
from torch.autograd import Variable

def squash(tensor: Variable) -> Variable:
    """Combine the first two dimensions into one dimension"""
    input_size = tensor.size()
    if len(input_size) <= 2:
        raise RuntimeError(f"No dimension to distribute: {input_size}")
    squashed_shape = [-1] + [x for x in input_size[2:]]
    return tensor.contiguous().view(*squashed_shape)


def unsquash(tensor: Variable, dim1: int, dim2: int) -> Variable:
    """Spread the first dimension into two dimensions"""
    new_shape = [dim1, dim2] + [dim for dim in tensor.size()[1:]]
    return tensor.contiguous().view(*new_shape)


class TimeDistributed(torch.nn.Module):
    """
    Given an input shaped like ``(batch_size, time_steps, [rest])`` and a ``Module`` that takes
    inputs like ``(batch_size, [rest])``, ``TimeDistributed`` reshapes the input to be
    ``(batch_size * time_steps, [rest])``, applies the contained ``Module``, then reshapes it back.

    Note that while the above gives shapes with ``batch_size`` first, this ``Module`` also works if
    ``batch_size`` is second - we always just combine the first two dimensions, then split them.
    """
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self._module = module

    def forward(self, *inputs):  # pylint: disable=arguments-differ
        reshaped_inputs = []
        for input_tensor in inputs:
            # Squash batch_size and time_steps into a single axis; result has shape
            # (batch_size * time_steps, input_size).
            squashed_tensor = squash(input_tensor)
            reshaped_inputs.append(squashed_tensor)

        reshaped_outputs = self._module(*reshaped_inputs)

        # Now get the output back into the right shape.
        # (batch_size, time_steps, [hidden_size])
        batch_size, time_steps, *_ = inputs[0].size()
        outputs = unsquash(reshaped_outputs, batch_size, time_steps)

        return outputs
