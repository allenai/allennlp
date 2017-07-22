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
    # Flip mask to make padded elements equal to one.
    inverse_mask = mask == 0
    # Number of padded elements in sequence.
    num_padded_elements = inverse_mask.sum(1)
    # Sequence length is max sequence length minus
    # the number of padded elements.
    length_indices = mask.size()[1] - num_padded_elements
    return length_indices.squeeze().long()


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
    Parameters
    ----------
    dropout_probability: float, Probability of dropping a dimension of the input.
    shape: Shape of the tensor you are generating a mask for.

    Return
    ------
    A torch.FloatTensor consisting of the binary mask scaled by 1/ (1 - dropout_probability).
    (this scaling ensures expected values and variances of the output of applying this mask
     and the original tensor are the same).
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
