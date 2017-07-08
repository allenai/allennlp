
import torch


def get_lengths_from_binary_sequence_mask(mask: torch.ByteTensor):

    # Flip mask to make padded elements equal to one.
    inverse_mask = mask == 0
    # Number of padded elements in sequence.
    num_padded_elements = inverse_mask.sum(1)
    # Sequence length is max sequence length minus
    # the number of padded elements.
    length_indices = mask.size()[1] - num_padded_elements
    return length_indices.squeeze().float()


def get_lengths_from_sequence_tensor(tensor: torch.FloatTensor):

    binary_mask = (tensor != 0.0).sum(-1) == tensor.size(-1)

    return get_lengths_from_binary_sequence_mask(binary_mask)


def sort_by_length(tensor, sequence_lengths):

    seq_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor[permutation_index]

    indices = torch.range(0, len(sequence_lengths) - 1)
    _, reverse_mapping = permutation_index.sort(0, descending=True)

    restoration_indices = indices[reverse_mapping]

    return sorted_tensor, restoration_indices

