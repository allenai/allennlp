from typing import Dict, List, Union
import torch
from torch.autograd import Variable
import numpy


def get_lengths_from_binary_sequence_mask(mask: torch.ByteTensor):
    """
    Compute sequence lengths for each batch element in a tensor using a
    binary mask.

    Parameters
    ----------
    mask : torch.ByteTensor, required.
        A 2D binary mask of shape (batch_size, sequence_length) to
        calculate the per-batch sequence lengths from.

    Returns
    -------
    A torch.LongTensor of shape (batch_size,) representing the lengths
    of the sequences in the batch.
    """
    return mask.sum(-1).squeeze()



def sort_batch_by_length(tensor: torch.FloatTensor, sequence_lengths: torch.LongTensor):
    """
    Sort a batch first tensor by some specified lengths.

    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A batch first Pytorch tensor.
    sequence_lengths : torch.LongTensor, required.
        A tensor representing the lengths of some dimension of the tensor which
        we want to sort by.

    Returns
    -------
    sorted_tensor : torch.FloatTensor
        The original tensor sorted along the batch dimension with respect to sequence_lengths.
    sorted_sequence_lengths : torch.LongTensor
        The original sequence_lengths sorted by decreasing size.
    restoration_indices : torch.LongTensor
        Indices into the sorted_tensor such that ``sorted_tensor[restoration_indices] == original_tensor``
    """
    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor[permutation_index]
    # This is the equivalent of zipping with index, sorting by the original
    # sequence lengths and returning the now sorted indices.
    index_range = torch.range(0, len(sequence_lengths) - 1).long()
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range[reverse_mapping]
    return sorted_tensor, sorted_sequence_lengths, restoration_indices


def get_dropout_mask(dropout_probability: float, shape: List[int]):
    """
    Computes an element-wise dropout mask for an arbitrarily sized tensor, where
    each element in the mask is dropped out with probability dropout_probability.

    Parameters
    ----------
    dropout_probability: float, Probability of dropping a dimension of the input.
    shape: Shape of the tensor you are generating a mask for.

    Return
    ------
    A torch.FloatTensor consisting of the binary mask scaled by 1/ (1 - dropout_probability).
    This scaling ensures expected values and variances of the output of applying this mask
     and the original tensor are the same.
    """
    binary_mask = torch.rand(shape) > dropout_probability
    # Scale mask by 1/keep_prob to preserve output statistics.
    dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
    return dropout_mask


def arrays_to_variables(data_structure: Dict[str, Union[dict, numpy.ndarray]],
                        cuda_device: int = -1):
    """
    Convert an (optionally) nested dictionary of arrays to Pytorch ``Variables``,
    suitable for use in a computation graph.
    """
    if isinstance(data_structure, dict):
        for key, value in data_structure.items():
            data_structure[key] = arrays_to_variables(value)
        return data_structure
    else:
        torch_variable = Variable(torch.from_numpy(data_structure))
        if cuda_device == -1:
            return torch_variable
        else:
            return torch_variable.cuda(cuda_device)


def masked_softmax(vector, mask):
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.

    We assume that both ``vector`` and ``mask`` (if given) have shape ``(batch_size, vector_dim)``.

    In the case that the input vector is completely masked, this function returns an array
    of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of a model
    that uses categorial cross-entropy loss.
    """
    # We calculate masked softmax in a numerically stable fashion, as done
    # in https://github.com/rkadlec/asreader/blob/master/asreader/custombricks/softmax_mask_bricks.py
    if mask is not None:
        # TODO(mattg): a bunch of this logic can be simplified once pytorch-0.2 is out.
        # torch.max(keepdim=True), for instance, simplifies things here.
        # Here we get normalized log probabilities for enhanced numerical stability.
        input_masked = mask * vector
        shifted = mask * (input_masked - torch.max(input_masked, dim=1)[0].expand_as(input_masked))
        # We add epsilon to avoid numerical instability when the sum in the log yields 0.
        normalization_constant = ((mask * shifted.exp()).sum(dim=1) + 1e-7).log()
        normalized_log_probabilities = (shifted - normalization_constant.expand_as(shifted))
        probabilities = normalized_log_probabilities.exp()
        return mask * probabilities
    else:
        # There is no mask, so we use the provided ``torch.nn.functional.softmax`` function.
        return torch.nn.functional.softmax(vector)
