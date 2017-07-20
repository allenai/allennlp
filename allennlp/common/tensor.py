from typing import List
import torch


def get_lengths_from_binary_sequence_mask(mask: torch.ByteTensor):
    """
    Compute sequence lengths for each batch element in a tensor using a
    binary mask.

    Parameters
    ----------
    mask : torch.ByteTensor, required
        A 2D binary mask of shape (batch_size, sequence_length) to
        calculate the per-batch sequence lengths from.

    Returns
    -------
    A torch.FloatTensor of shape (batch_size,) representing the lengths
    of the sequences in the batch.
    """

    # Flip mask to make padded elements equal to one.
    inverse_mask = mask == 0
    # Number of padded elements in sequence.
    num_padded_elements = inverse_mask.sum(1)
    # Sequence length is max sequence length minus
    # the number of padded elements.
    length_indices = mask.size()[1] - num_padded_elements
    return length_indices.squeeze().float()


def sort_batch_by_length(tensor: torch.FloatTensor, sequence_lengths: torch.LongTensor):
    """
    Sort a batch first tensor by some specified lengths.

    Parameters
    ----------
    tensor : torch.FloatTensor, required
        A batch first Pytorch tensor.
    sequence_lengths : torch.LongTensor, required.
        A tensor representing the lengths of some dimension of the tensor which
        we want to sort by.

    Returns
    -------
    sorted_tensor : torch.FloatTensor
        The original tensor sorted along the batch dimension with respect to sequence_lengths.
    restoration_indices : torch.LongTensor
        Indices into the sorted_tensor such that ``sorted_tensor[restoration_indices] == original_tensor``
    """

    seq_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor[permutation_index]

    # This is the equivalent of zipping with index, sorting
    index_range = torch.range(0, len(sequence_lengths) - 1).long()
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range[reverse_mapping]

    return sorted_tensor, restoration_indices


def get_dropout_mask(dropout_probability: float, shape: List[int]):
    """
    Parameters
    ----------
    dropout_probability: float, Probability of dropping a dimension of the input.
    shape: Shape of the tensor you are generating a mask for.

    Return
    ------
    Float tensor consisting of the binary mask scaled by 1/ (1 - dropout_probability).
    (this scaling ensures expected values and variances of the output of
     applying this mask and the original tensor are the same).
    """
    binary_mask = torch.rand(shape) > dropout_probability
    # Scale mask by 1/keep_prob to preserve output statistics.
    dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
    return dropout_mask