
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

    seq_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor[permutation_index]

    indices = torch.range(0, len(sequence_lengths) - 1)
    _, reverse_mapping = permutation_index.sort(0, descending=True)

    restoration_indices = indices[reverse_mapping]

    return sorted_tensor, restoration_indices

