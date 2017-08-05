from typing import Dict, List, Optional, Union

import numpy
import torch
from torch.autograd import Variable

from allennlp.common.checks import ConfigurationError


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


def sort_batch_by_length(tensor: torch.autograd.Variable, sequence_lengths: torch.autograd.Variable):
    """
    Sort a batch first tensor by some specified lengths.

    Parameters
    ----------
    tensor : Variable(torch.FloatTensor), required.
        A batch first Pytorch tensor.
    sequence_lengths : Variable(torch.LongTensor), required.
        A tensor representing the lengths of some dimension of the tensor which
        we want to sort by.

    Returns
    -------
    sorted_tensor : Variable(torch.FloatTensor)
        The original tensor sorted along the batch dimension with respect to sequence_lengths.
    sorted_sequence_lengths : Variable(torch.LongTensor)
        The original sequence_lengths sorted by decreasing size.
    restoration_indices : Variable(torch.LongTensor)
        Indices into the sorted_tensor such that
        ``sorted_tensor.index_select(0, restoration_indices) == original_tensor``
    """

    if not isinstance(tensor, Variable) or not isinstance(sequence_lengths, Variable):
        raise ConfigurationError("Both the tensor and sequence lengths must be torch.autograd.Variables.")

    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)
    # This is the equivalent of zipping with index, sorting by the original
    # sequence lengths and returning the now sorted indices.
    index_range = Variable(torch.arange(0, len(sequence_lengths)).long())
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
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
                        cuda_device: int = -1,
                        add_batch_dimension: bool = False):
    """
    Convert an (optionally) nested dictionary of arrays to Pytorch ``Variables``,
    suitable for use in a computation graph.

    Parameters
    ----------
    data_structure : Dict[str, Union[dict, numpy.ndarray]], required.
        The nested dictionary of arrays to convert to Pytorch ``Variables``.
    cuda_device : int, optional (default = -1)
        If cuda_device <= 0, GPUs are available and Pytorch was compiled with
        CUDA support, the tensor will be copied to the cuda_device specified.
    add_batch_dimension : bool, optional (default = False).
        Optionally add a batch dimension to tensors converted to ``Variables``
        using this function. This is useful during inference for passing
        tensors representing a single example to a Pytorch model which
        would otherwise not have a batch dimension.

    Returns
    -------
    The original data structure or tensor converted to a Pytorch ``Variable``.
    """
    if isinstance(data_structure, dict):
        for key, value in data_structure.items():
            data_structure[key] = arrays_to_variables(value, cuda_device, add_batch_dimension)
        return data_structure
    else:
        tensor = torch.from_numpy(data_structure)
        if add_batch_dimension:
            tensor.unsqueeze_(0)
        torch_variable = Variable(tensor)
        if cuda_device == -1:
            return torch_variable
        else:
            return torch_variable.cuda(cuda_device)


def _get_normalized_masked_log_probablities(vector, mask):
    # We calculate normalized log probabilities in a numerically stable fashion, as done
    # in https://github.com/rkadlec/asreader/blob/master/asreader/custombricks/softmax_mask_bricks.py
    # TODO(mattg): a bunch of this logic can be simplified once pytorch-0.2 is out.
    # torch.max(keepdim=True), for instance, simplifies things here.
    input_masked = mask * vector
    shifted = mask * (input_masked - torch.max(input_masked, dim=1)[0].expand_as(input_masked))
    # We add epsilon to avoid numerical instability when the sum in the log yields 0.
    normalization_constant = ((mask * shifted.exp()).sum(dim=1) + 1e-7).log()
    normalized_log_probabilities = (shifted - normalization_constant.expand_as(shifted))
    return normalized_log_probabilities


def masked_softmax(vector, mask):
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.

    We assume that both ``vector`` and ``mask`` (if given) have shape ``(batch_size, vector_dim)``.

    In the case that the input vector is completely masked, this function returns an array
    of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of a model
    that uses categorical cross-entropy loss.
    """
    if mask is not None:
        return mask * _get_normalized_masked_log_probablities(vector, mask).exp()
    else:
        # There is no mask, so we use the provided ``torch.nn.functional.softmax`` function.
        return torch.nn.functional.softmax(vector)


def masked_log_softmax(vector, mask):
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.

    We assume that both ``vector`` and ``mask`` (if given) have shape ``(batch_size, vector_dim)``.

    In the case that the input vector is completely masked, this function returns an array
    of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of a model
    that uses categorical cross-entropy loss.
    """
    if mask is not None:
        return mask * _get_normalized_masked_log_probablities(vector, mask)
    else:
        # There is no mask, so we use the provided ``torch.nn.functional.log_softmax`` function.
        return torch.nn.functional.log_softmax(vector)


def viterbi_decode(tag_sequence: torch.Tensor, transition_matrix: torch.Tensor):
    """
    Perform Viterbi decoding in log space over a sequence given a transition matrix
    specifying pairwise (transition) potentials between tags and a matrix of shape
    (sequence_length, num_tags) specifying unary potentials for possible tags per
    timestep.

    Parameters
    ----------
    tag_sequence : torch.Tensor, required.
        A tensor of shape (sequence_length, num_tags) representing scores for
        a set of tags over a given sequence.
    transition_matrix : torch.Tensor, required.
        A tensor of shape (num_tags, num_tags) representing the binary potentials
        for transitioning between a given pair of tags.

    Returns
    -------
    viterbi_path : List[int]
        The tag indices of the maximum likelihood tag sequence.
    viterbi_score : float
        The score of the viterbi path.
    """
    sequence_length, _ = list(tag_sequence.size())
    path_scores = []
    path_indices = []
    path_scores.append(tag_sequence[0, :])

    # Evaluate the scores for all possible paths.
    for timestep in range(1, sequence_length):
        # TODO(Mark): Use broadcasting here once Pytorch 0.2 is released.
        tiled_path_scores = path_scores[timestep - 1].expand_as(transition_matrix).transpose(0, 1)
        summed_potentials = tiled_path_scores + transition_matrix
        scores, paths = torch.max(summed_potentials, 0)
        path_scores.append(tag_sequence[timestep, :] + scores.squeeze())
        path_indices.append(paths.squeeze())

    # Construct the most likely sequence backwards.
    viterbi_score, best_path = torch.max(path_scores[-1], 0)
    viterbi_path = [int(best_path.numpy())]
    for backward_timestep in reversed(path_indices):
        viterbi_path.append(int(backward_timestep[viterbi_path[-1]]))
    # Reverse the backward path.
    viterbi_path.reverse()
    return viterbi_path, viterbi_score


def get_text_field_mask(text_field_tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Takes the dictionary of tensors produced by a ``TextField`` and returns a mask of shape
    ``(batch_size, num_tokens)``.  This mask will be 0 where the tokens are padding, and 1
    otherwise.

    There could be several entries in the tensor dictionary with different shapes (e.g., one for
    word ids, one for character ids).  In order to get a token mask, we assume that the tensor in
    the dictionary with the lowest number of dimensions has plain token ids.  This allows us to
    also handle cases where the input is actually a ``ListField[TextField]``.
    """
    tensor_dims = [(tensor.dim(), tensor) for tensor in text_field_tensors.values()]
    tensor_dims.sort(key=lambda x: x[0])
    token_tensor = tensor_dims[0][1]
    return token_tensor != 0


def last_dim_softmax(tensor: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Takes a tensor with 3 or more dimensions and does a masked softmax over the last dimension.  We
    assume the tensor has shape ``(batch_size, ..., sequence_length)`` and that the mask (if given)
    has shape ``(batch_size, sequence_length)``.  We first unsqueeze and expand the mask so that it
    has the same shape as the tensor, then flatten them both to be 2D, pass them through
    :func:`masked_softmax`, then put the tensor back in its original shape.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor.size()[-1])
    if mask:
        while mask.dim() < tensor.dim():
            mask = mask.unsqueeze(1)
        mask = mask.expand_as(tensor).contiguous().float()
        mask = mask.view(-1, mask.size()[-1])
    reshaped_result = masked_softmax(reshaped_tensor, mask)
    return reshaped_result.view(*tensor_shape)


def weighted_sum(matrix: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    """
    Takes a matrix of vectors and a set of weights over the rows in the matrix (which we call an
    "attention" vector), and returns a weighted sum of the rows in the matrix.  This is the typical
    computation performed after an attention mechanism.

    Note that while we call this a "matrix" of vectors and an attention "vector", we also handle
    higher-order tensors.  We always sum over the second-to-last dimension of the "matrix", and we
    assume that all dimensions in the "matrix" prior to the last dimension are matched in the
    "vector".  Non-matched dimensions in the "vector" must be `directly after the batch dimension`.

    For example, say I have a "matrix" with dimensions ``(batch_size, num_queries, num_words,
    embedding_dim)``.  The attention "vector" then must have at least those dimensions, and could
    have more. Both:

        - ``(batch_size, num_queries, num_words)`` (distribution over words for each query)
        - ``(batch_size, num_documents, num_queries, num_words)`` (distribution over words in a
          query for each document)

    are valid input "vectors", producing tensors of shape:
    ``(batch_size, num_queries, embedding_dim)`` and
    ``(batch_size, num_documents, num_queries, embedding_dim)`` respectively.
    """
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2).squeeze(-2)


def sequence_cross_entropy_with_logits(logits: torch.FloatTensor,
                                       targets: torch.LongTensor,
                                       weights: torch.FloatTensor,
                                       batch_average: bool = True) -> torch.FloatTensor:
    """
    Computes the cross entropy loss of a sequence, weighted with respect to
    some user provided weights. Note that the weighting here is not the same as
    in the :func:`torch.nn.CrossEntropyLoss()` criterion, which is weighting
    classes; here we are weighting the loss contribution from particular elements
    in the sequence. This allows loss computations for models which use padding.

    Parameters
    ----------
    logits : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch_size, sequence_length, num_classes)
        which contains the unnormalized probability for each class.
    targets : ``torch.LongTensor``, required.
        A ``torch.LongTensor`` of size (batch, sequence_length) which contains the
        index of the true class for each corresponding step.
    weights : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch, sequence_length)
    batch_average : bool, optional, (default = True).
        A bool indicating whether the loss should be averaged across the batch,
        or returned as a vector of losses per batch element.

    Returns
    -------
    A torch.FloatTensor representing the cross entropy loss.
    If ``batch_average == True``, the returned loss is a scalar.
    If ``batch_average == False``, the returned loss is a vector of shape (batch_size,).

    """
    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    # Contribution to the negative log likelihood only comes from the exact indices
    # of the targets, as the target distributions are one-hot. Here we use torch.gather
    # to extract the indices of the num_classes dimension which contribute to the loss.
    # shape : (batch * sequence_length, 1)
    negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights.float()
    # shape : (batch_size,)
    per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)

    if batch_average:
        num_non_empty_sequences = ((weights.sum(1) > 0).float().sum() + 1e-13)
        return per_batch_loss.sum() / num_non_empty_sequences
    return per_batch_loss.squeeze(1)


def replace_masked_values(tensor: Variable, mask: Variable, replace_with: float) -> Variable:
    """
    Replaces all masked values in ``tensor`` with ``replace_with``.  ``mask`` must be broadcastable
    to the same shape as ``tensor``.

    Until pytorch 0.2 is released, we require that ``tensor.dim() == mask.dim()``.  Otherwise we
    won't know which dimensions of the mask to unsqueeze.
    """
    # We'll build a tensor of the same shape as `tensor`, subtract away masked values, then add
    # back in the `replace_with` value.
    # TODO(mattg): use broadcasting here when it's available in pytorch 0.2.
    if tensor.dim() != mask.dim():
        raise ConfigurationError("tensor.dim() (%d) != mask.dim() (%d)" % (tensor.dim(), mask.dim()))
    one_minus_mask = 1.0 - mask.expand_as(tensor)
    values_to_subtract = tensor * one_minus_mask
    values_to_add = replace_with * one_minus_mask
    return tensor - values_to_subtract + values_to_add
