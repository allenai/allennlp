"""
Assorted utilities for working with neural networks in AllenNLP.
"""
# pylint: disable=too-many-lines
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar
import logging
import copy
import math

import torch

from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

T = TypeVar('T')


def has_tensor(obj) -> bool:
    """
    Given a possibly complex data structure,
    check if it has any torch.Tensors in it.
    """
    if isinstance(obj, torch.Tensor):
        return True
    elif isinstance(obj, dict):
        return any(has_tensor(value) for value in obj.values())
    elif isinstance(obj, (list, tuple)):
        return any(has_tensor(item) for item in obj)
    else:
        return False


def move_to_device(obj, cuda_device: int):
    """
    Given a structure (possibly) containing Tensors on the CPU,
    move all the Tensors to the specified GPU (or do nothing, if they should be on the CPU).
    """
    if cuda_device < 0 or not has_tensor(obj):
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.cuda(cuda_device)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, cuda_device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, cuda_device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple([move_to_device(item, cuda_device) for item in obj])
    else:
        return obj


def clamp_tensor(tensor, minimum, maximum):
    """
    Supports sparse and dense tensors.
    Returns a tensor with values clamped between the provided minimum and maximum,
    without modifying the original tensor.
    """
    if tensor.is_sparse:
        coalesced_tensor = tensor.coalesce()
        # pylint: disable=protected-access
        coalesced_tensor._values().clamp_(minimum, maximum)
        return coalesced_tensor
    else:
        return tensor.clamp(minimum, maximum)


def batch_tensor_dicts(tensor_dicts: List[Dict[str, torch.Tensor]],
                       remove_trailing_dimension: bool = False) -> Dict[str, torch.Tensor]:
    """
    Takes a list of tensor dictionaries, where each dictionary is assumed to have matching keys,
    and returns a single dictionary with all tensors with the same key batched together.

    Parameters
    ----------
    tensor_dicts : ``List[Dict[str, torch.Tensor]]``
        The list of tensor dictionaries to batch.
    remove_trailing_dimension : ``bool``
        If ``True``, we will check for a trailing dimension of size 1 on the tensors that are being
        batched, and remove it if we find it.
    """
    key_to_tensors: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for tensor_dict in tensor_dicts:
        for key, tensor in tensor_dict.items():
            key_to_tensors[key].append(tensor)
    batched_tensors = {}
    for key, tensor_list in key_to_tensors.items():
        batched_tensor = torch.stack(tensor_list)
        if remove_trailing_dimension and all(tensor.size(-1) == 1 for tensor in tensor_list):
            batched_tensor = batched_tensor.squeeze(-1)
        batched_tensors[key] = batched_tensor
    return batched_tensors


def get_lengths_from_binary_sequence_mask(mask: torch.Tensor):
    """
    Compute sequence lengths for each batch element in a tensor using a
    binary mask.

    Parameters
    ----------
    mask : torch.Tensor, required.
        A 2D binary mask of shape (batch_size, sequence_length) to
        calculate the per-batch sequence lengths from.

    Returns
    -------
    A torch.LongTensor of shape (batch_size,) representing the lengths
    of the sequences in the batch.
    """
    return mask.long().sum(-1)


def get_mask_from_sequence_lengths(sequence_lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """
    Given a variable of shape ``(batch_size,)`` that represents the sequence lengths of each batch
    element, this function returns a ``(batch_size, max_length)`` mask variable.  For example, if
    our input was ``[2, 2, 3]``, with a ``max_length`` of 4, we'd return
    ``[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]``.

    We require ``max_length`` here instead of just computing it from the input ``sequence_lengths``
    because it lets us avoid finding the max, then copying that value from the GPU to the CPU so
    that we can use it to construct a new tensor.
    """
    # (batch_size, max_length)
    ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return (sequence_lengths.unsqueeze(1) >= range_tensor).long()


def sort_batch_by_length(tensor: torch.Tensor, sequence_lengths: torch.Tensor):
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
        Indices into the sorted_tensor such that
        ``sorted_tensor.index_select(0, restoration_indices) == original_tensor``
    permuation_index : torch.LongTensor
        The indices used to sort the tensor. This is useful if you want to sort many
        tensors using the same ordering.
    """

    if not isinstance(tensor, torch.Tensor) or not isinstance(sequence_lengths, torch.Tensor):
        raise ConfigurationError("Both the tensor and sequence lengths must be torch.Tensors.")

    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)

    index_range = torch.arange(0, len(sequence_lengths), device=sequence_lengths.device)
    # This is the equivalent of zipping with index, sorting by the original
    # sequence lengths and returning the now sorted indices.
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index


def get_final_encoder_states(encoder_outputs: torch.Tensor,
                             mask: torch.Tensor,
                             bidirectional: bool = False) -> torch.Tensor:
    """
    Given the output from a ``Seq2SeqEncoder``, with shape ``(batch_size, sequence_length,
    encoding_dim)``, this method returns the final hidden state for each element of the batch,
    giving a tensor of shape ``(batch_size, encoding_dim)``.  This is not as simple as
    ``encoder_outputs[:, -1]``, because the sequences could have different lengths.  We use the
    mask (which has shape ``(batch_size, sequence_length)``) to find the final state for each batch
    instance.

    Additionally, if ``bidirectional`` is ``True``, we will split the final dimension of the
    ``encoder_outputs`` into two and assume that the first half is for the forward direction of the
    encoder and the second half is for the backward direction.  We will concatenate the last state
    for each encoder dimension, giving ``encoder_outputs[:, -1, :encoding_dim/2]`` concatenated with
    ``encoder_outputs[:, 0, encoding_dim/2:]``.
    """
    # These are the indices of the last words in the sequences (i.e. length sans padding - 1).  We
    # are assuming sequences are right padded.
    # Shape: (batch_size,)
    last_word_indices = mask.sum(1).long() - 1
    batch_size, _, encoder_output_dim = encoder_outputs.size()
    expanded_indices = last_word_indices.view(-1, 1, 1).expand(batch_size, 1, encoder_output_dim)
    # Shape: (batch_size, 1, encoder_output_dim)
    final_encoder_output = encoder_outputs.gather(1, expanded_indices)
    final_encoder_output = final_encoder_output.squeeze(1)  # (batch_size, encoder_output_dim)
    if bidirectional:
        final_forward_output = final_encoder_output[:, :(encoder_output_dim // 2)]
        final_backward_output = encoder_outputs[:, 0, (encoder_output_dim // 2):]
        final_encoder_output = torch.cat([final_forward_output, final_backward_output], dim=-1)
    return final_encoder_output


def get_dropout_mask(dropout_probability: float, tensor_for_masking: torch.Tensor):
    """
    Computes and returns an element-wise dropout mask for a given tensor, where
    each element in the mask is dropped out with probability dropout_probability.
    Note that the mask is NOT applied to the tensor - the tensor is passed to retain
    the correct CUDA tensor type for the mask.

    Parameters
    ----------
    dropout_probability : float, required.
        Probability of dropping a dimension of the input.
    tensor_for_masking : torch.Tensor, required.


    Returns
    -------
    A torch.FloatTensor consisting of the binary mask scaled by 1/ (1 - dropout_probability).
    This scaling ensures expected values and variances of the output of applying this mask
     and the original tensor are the same.
    """
    binary_mask = (torch.rand(tensor_for_masking.size()) > dropout_probability).to(tensor_for_masking.device)
    # Scale mask by 1/keep_prob to preserve output statistics.
    dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
    return dropout_mask


def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.

    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.

    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.

    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.

    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


def masked_max(vector: torch.Tensor,
               mask: torch.Tensor,
               dim: int,
               keepdim: bool = False,
               min_val: float = -1e7) -> torch.Tensor:
    """
    To calculate max along certain dimensions on masked values

    Parameters
    ----------
    vector : ``torch.Tensor``
        The vector to calculate max, assume unmasked parts are already zeros
    mask : ``torch.Tensor``
        The mask of the vector. It must be broadcastable with vector.
    dim : ``int``
        The dimension to calculate max
    keepdim : ``bool``
        Whether to keep dimension
    min_val : ``float``
        The minimal value for paddings

    Returns
    -------
    A ``torch.Tensor`` of including the maximum values.
    """
    one_minus_mask = (1.0 - mask).byte()
    replaced_vector = vector.masked_fill(one_minus_mask, min_val)
    max_value, _ = replaced_vector.max(dim=dim, keepdim=keepdim)
    return max_value


def masked_mean(vector: torch.Tensor,
                mask: torch.Tensor,
                dim: int,
                keepdim: bool = False,
                eps: float = 1e-8) -> torch.Tensor:
    """
    To calculate mean along certain dimensions on masked values

    Parameters
    ----------
    vector : ``torch.Tensor``
        The vector to calculate mean.
    mask : ``torch.Tensor``
        The mask of the vector. It must be broadcastable with vector.
    dim : ``int``
        The dimension to calculate mean
    keepdim : ``bool``
        Whether to keep dimension
    eps : ``float``
        A small value to avoid zero division problem.

    Returns
    -------
    A ``torch.Tensor`` of including the mean values.
    """
    one_minus_mask = (1.0 - mask).byte()
    replaced_vector = vector.masked_fill(one_minus_mask, 0.0)

    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask.float(), dim=dim, keepdim=keepdim)
    return value_sum / value_count.clamp(min=eps)


def masked_flip(padded_sequence: torch.Tensor,
                sequence_lengths: List[int]) -> torch.Tensor:
    """
        Flips a padded tensor along the time dimension without affecting masked entries.

        Parameters
        ----------
        padded_sequence : ``torch.Tensor``
            The tensor to flip along the time dimension.
            Assumed to be of dimensions (batch size, num timesteps, ...)
        sequence_lengths : ``torch.Tensor``
            A list containing the lengths of each unpadded sequence in the batch.

        Returns
        -------
        A ``torch.Tensor`` of the same shape as padded_sequence.
        """
    assert padded_sequence.size(0) == len(sequence_lengths), \
        f'sequence_lengths length ${len(sequence_lengths)} does not match batch size ${padded_sequence.size(0)}'
    num_timesteps = padded_sequence.size(1)
    flipped_padded_sequence = torch.flip(padded_sequence, [1])
    sequences = [flipped_padded_sequence[i, num_timesteps - length:] for i, length in enumerate(sequence_lengths)]
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)


def viterbi_decode(tag_sequence: torch.Tensor,
                   transition_matrix: torch.Tensor,
                   tag_observations: Optional[List[int]] = None):
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
    tag_observations : Optional[List[int]], optional, (default = None)
        A list of length ``sequence_length`` containing the class ids of observed
        elements in the sequence, with unobserved elements being set to -1. Note that
        it is possible to provide evidence which results in degenerate labelings if
        the sequences of tags you provide as evidence cannot transition between each
        other, or those transitions are extremely unlikely. In this situation we log a
        warning, but the responsibility for providing self-consistent evidence ultimately
        lies with the user.

    Returns
    -------
    viterbi_path : List[int]
        The tag indices of the maximum likelihood tag sequence.
    viterbi_score : torch.Tensor
        The score of the viterbi path.
    """
    sequence_length, num_tags = list(tag_sequence.size())
    if tag_observations:
        if len(tag_observations) != sequence_length:
            raise ConfigurationError("Observations were provided, but they were not the same length "
                                     "as the sequence. Found sequence of length: {} and evidence: {}"
                                     .format(sequence_length, tag_observations))
    else:
        tag_observations = [-1 for _ in range(sequence_length)]

    path_scores = []
    path_indices = []

    if tag_observations[0] != -1:
        one_hot = torch.zeros(num_tags)
        one_hot[tag_observations[0]] = 100000.
        path_scores.append(one_hot)
    else:
        path_scores.append(tag_sequence[0, :])

    # Evaluate the scores for all possible paths.
    for timestep in range(1, sequence_length):
        # Add pairwise potentials to current scores.
        summed_potentials = path_scores[timestep - 1].unsqueeze(-1) + transition_matrix
        scores, paths = torch.max(summed_potentials, 0)

        # If we have an observation for this timestep, use it
        # instead of the distribution over tags.
        observation = tag_observations[timestep]
        # Warn the user if they have passed
        # invalid/extremely unlikely evidence.
        if tag_observations[timestep - 1] != -1:
            if transition_matrix[tag_observations[timestep - 1], observation] < -10000:
                logger.warning("The pairwise potential between tags you have passed as "
                               "observations is extremely unlikely. Double check your evidence "
                               "or transition potentials!")
        if observation != -1:
            one_hot = torch.zeros(num_tags)
            one_hot[observation] = 100000.
            path_scores.append(one_hot)
        else:
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


def get_text_field_mask(text_field_tensors: Dict[str, torch.Tensor],
                        num_wrapping_dims: int = 0) -> torch.LongTensor:
    """
    Takes the dictionary of tensors produced by a ``TextField`` and returns a mask
    with 0 where the tokens are padding, and 1 otherwise.  We also handle ``TextFields``
    wrapped by an arbitrary number of ``ListFields``, where the number of wrapping ``ListFields``
    is given by ``num_wrapping_dims``.

    If ``num_wrapping_dims == 0``, the returned mask has shape ``(batch_size, num_tokens)``.
    If ``num_wrapping_dims > 0`` then the returned mask has ``num_wrapping_dims`` extra
    dimensions, so the shape will be ``(batch_size, ..., num_tokens)``.

    There could be several entries in the tensor dictionary with different shapes (e.g., one for
    word ids, one for character ids).  In order to get a token mask, we use the tensor in
    the dictionary with the lowest number of dimensions.  After subtracting ``num_wrapping_dims``,
    if this tensor has two dimensions we assume it has shape ``(batch_size, ..., num_tokens)``,
    and use it for the mask.  If instead it has three dimensions, we assume it has shape
    ``(batch_size, ..., num_tokens, num_features)``, and sum over the last dimension to produce
    the mask.  Most frequently this will be a character id tensor, but it could also be a
    featurized representation of each token, etc.

    If the input ``text_field_tensors`` contains the "mask" key, this is returned instead of inferring the mask.

    TODO(joelgrus): can we change this?
    NOTE: Our functions for generating masks create torch.LongTensors, because using
    torch.ByteTensors  makes it easy to run into overflow errors
    when doing mask manipulation, such as summing to get the lengths of sequences - see below.
    >>> mask = torch.ones([260]).byte()
    >>> mask.sum() # equals 260.
    >>> var_mask = torch.autograd.V(mask)
    >>> var_mask.sum() # equals 4, due to 8 bit precision - the sum overflows.
    """
    if "mask" in text_field_tensors:
        return text_field_tensors["mask"]

    tensor_dims = [(tensor.dim(), tensor) for tensor in text_field_tensors.values()]
    tensor_dims.sort(key=lambda x: x[0])

    smallest_dim = tensor_dims[0][0] - num_wrapping_dims
    if smallest_dim == 2:
        token_tensor = tensor_dims[0][1]
        return (token_tensor != 0).long()
    elif smallest_dim == 3:
        character_tensor = tensor_dims[0][1]
        return ((character_tensor > 0).long().sum(dim=-1) > 0).long()
    else:
        raise ValueError("Expected a tensor with dimension 2 or 3, found {}".format(smallest_dim))


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
    # We'll special-case a few settings here, where there are efficient (but poorly-named)
    # operations in pytorch that already do the computation we need.
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)


def sequence_cross_entropy_with_logits(logits: torch.FloatTensor,
                                       targets: torch.LongTensor,
                                       weights: torch.FloatTensor,
                                       average: str = "batch",
                                       label_smoothing: float = None) -> torch.FloatTensor:
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
    average: str, optional (default = "batch")
        If "batch", average the loss across the batches. If "token", average
        the loss across each item in the input. If ``None``, return a vector
        of losses per batch element.
    label_smoothing : ``float``, optional (default = None)
        Whether or not to apply label smoothing to the cross-entropy loss.
        For example, with a label smoothing value of 0.2, a 4 class classification
        target would look like ``[0.05, 0.05, 0.85, 0.05]`` if the 3rd class was
        the correct label.

    Returns
    -------
    A torch.FloatTensor representing the cross entropy loss.
    If ``average=="batch"`` or ``average=="token"``, the returned loss is a scalar.
    If ``average is None``, the returned loss is a vector of shape (batch_size,).

    """
    if average not in {None, "token", "batch"}:
        raise ValueError("Got average f{average}, expected one of "
                         "None, 'token', or 'batch'")

    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / num_classes
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = - log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # Contribution to the negative log likelihood only comes from the exact indices
        # of the targets, as the target distributions are one-hot. Here we use torch.gather
        # to extract the indices of the num_classes dimension which contribute to the loss.
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights.float()

    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        num_non_empty_sequences = ((weights.sum(1) > 0).float().sum() + 1e-13)
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (weights.sum().float() + 1e-13)
    else:
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        return per_batch_loss


def replace_masked_values(tensor: torch.Tensor, mask: torch.Tensor, replace_with: float) -> torch.Tensor:
    """
    Replaces all masked values in ``tensor`` with ``replace_with``.  ``mask`` must be broadcastable
    to the same shape as ``tensor``. We require that ``tensor.dim() == mask.dim()``, as otherwise we
    won't know which dimensions of the mask to unsqueeze.

    This just does ``tensor.masked_fill()``, except the pytorch method fills in things with a mask
    value of 1, where we want the opposite.  You can do this in your own code with
    ``tensor.masked_fill((1 - mask).byte(), replace_with)``.
    """
    if tensor.dim() != mask.dim():
        raise ConfigurationError("tensor.dim() (%d) != mask.dim() (%d)" % (tensor.dim(), mask.dim()))
    return tensor.masked_fill((1 - mask).byte(), replace_with)


def tensors_equal(tensor1: torch.Tensor, tensor2: torch.Tensor, tolerance: float = 1e-12) -> bool:
    """
    A check for tensor equality (by value).  We make sure that the tensors have the same shape,
    then check all of the entries in the tensor for equality.  We additionally allow the input
    tensors to be lists or dictionaries, where we then do the above check on every position in the
    list / item in the dictionary.  If we find objects that aren't tensors as we're doing that, we
    just defer to their equality check.

    This is kind of a catch-all method that's designed to make implementing ``__eq__`` methods
    easier, in a way that's really only intended to be useful for tests.
    """
    # pylint: disable=too-many-return-statements
    if isinstance(tensor1, (list, tuple)):
        if not isinstance(tensor2, (list, tuple)) or len(tensor1) != len(tensor2):
            return False
        return all([tensors_equal(t1, t2, tolerance) for t1, t2 in zip(tensor1, tensor2)])
    elif isinstance(tensor1, dict):
        if not isinstance(tensor2, dict):
            return False
        if tensor1.keys() != tensor2.keys():
            return False
        return all([tensors_equal(tensor1[key], tensor2[key], tolerance) for key in tensor1])
    elif isinstance(tensor1, torch.Tensor):
        if not isinstance(tensor2, torch.Tensor):
            return False
        if tensor1.size() != tensor2.size():
            return False
        return ((tensor1 - tensor2).abs().float() < tolerance).all()
    else:
        try:
            return tensor1 == tensor2
        except RuntimeError:
            print(type(tensor1), type(tensor2))
            raise


def device_mapping(cuda_device: int):
    """
    In order to `torch.load()` a GPU-trained model onto a CPU (or specific GPU),
    you have to supply a `map_location` function. Call this with
    the desired `cuda_device` to get the function that `torch.load()` needs.
    """

    def inner_device_mapping(storage: torch.Storage, location) -> torch.Storage:  # pylint: disable=unused-argument
        if cuda_device >= 0:
            return storage.cuda(cuda_device)
        else:
            return storage

    return inner_device_mapping


def combine_tensors(combination: str, tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Combines a list of tensors using element-wise operations and concatenation, specified by a
    ``combination`` string.  The string refers to (1-indexed) positions in the input tensor list,
    and looks like ``"1,2,1+2,3-1"``.

    We allow the following kinds of combinations: ``x``, ``x*y``, ``x+y``, ``x-y``, and ``x/y``,
    where ``x`` and ``y`` are positive integers less than or equal to ``len(tensors)``.  Each of
    the binary operations is performed elementwise.  You can give as many combinations as you want
    in the ``combination`` string.  For example, for the input string ``"1,2,1*2"``, the result
    would be ``[1;2;1*2]``, as you would expect, where ``[;]`` is concatenation along the last
    dimension.

    If you have a fixed, known way to combine tensors that you use in a model, you should probably
    just use something like ``torch.cat([x_tensor, y_tensor, x_tensor * y_tensor])``.  This
    function adds some complexity that is only necessary if you want the specific combination used
    to be `configurable`.

    If you want to do any element-wise operations, the tensors involved in each element-wise
    operation must have the same shape.

    This function also accepts ``x`` and ``y`` in place of ``1`` and ``2`` in the combination
    string.
    """
    if len(tensors) > 9:
        raise ConfigurationError("Double-digit tensor lists not currently supported")
    combination = combination.replace('x', '1').replace('y', '2')
    to_concatenate = [_get_combination(piece, tensors) for piece in combination.split(',')]
    return torch.cat(to_concatenate, dim=-1)


def _rindex(sequence: Sequence[T], obj: T) -> int:
    """
    Return zero-based index in the sequence of the last item whose value is equal to obj.  Raises a
    ValueError if there is no such item.

    Parameters
    ----------
    sequence : ``Sequence[T]``
    obj : ``T``

    Returns
    -------
    zero-based index associated to the position of the last item equal to obj
    """
    for i in range(len(sequence) - 1, -1, -1):
        if sequence[i] == obj:
            return i

    raise ValueError(f"Unable to find {obj} in sequence {sequence}.")


def _get_combination(combination: str, tensors: List[torch.Tensor]) -> torch.Tensor:
    if combination.isdigit():
        index = int(combination) - 1
        return tensors[index]
    else:
        if len(combination) != 3:
            raise ConfigurationError("Invalid combination: " + combination)
        first_tensor = _get_combination(combination[0], tensors)
        second_tensor = _get_combination(combination[2], tensors)
        operation = combination[1]
        if operation == '*':
            return first_tensor * second_tensor
        elif operation == '/':
            return first_tensor / second_tensor
        elif operation == '+':
            return first_tensor + second_tensor
        elif operation == '-':
            return first_tensor - second_tensor
        else:
            raise ConfigurationError("Invalid operation: " + operation)


def combine_tensors_and_multiply(combination: str,
                                 tensors: List[torch.Tensor],
                                 weights: torch.nn.Parameter) -> torch.Tensor:
    """
    Like :func:`combine_tensors`, but does a weighted (linear) multiplication while combining.
    This is a separate function from ``combine_tensors`` because we try to avoid instantiating
    large intermediate tensors during the combination, which is possible because we know that we're
    going to be multiplying by a weight vector in the end.

    Parameters
    ----------
    combination : ``str``
        Same as in :func:`combine_tensors`
    tensors : ``List[torch.Tensor]``
        A list of tensors to combine, where the integers in the ``combination`` are (1-indexed)
        positions in this list of tensors.  These tensors are all expected to have either three or
        four dimensions, with the final dimension being an embedding.  If there are four
        dimensions, one of them must have length 1.
    weights : ``torch.nn.Parameter``
        A vector of weights to use for the combinations.  This should have shape (combined_dim,),
        as calculated by :func:`get_combined_dim`.
    """
    if len(tensors) > 9:
        raise ConfigurationError("Double-digit tensor lists not currently supported")
    combination = combination.replace('x', '1').replace('y', '2')
    pieces = combination.split(',')
    tensor_dims = [tensor.size(-1) for tensor in tensors]
    combination_dims = [_get_combination_dim(piece, tensor_dims) for piece in pieces]
    dims_so_far = 0
    to_sum = []
    for piece, combination_dim in zip(pieces, combination_dims):
        weight = weights[dims_so_far:(dims_so_far + combination_dim)]
        dims_so_far += combination_dim
        to_sum.append(_get_combination_and_multiply(piece, tensors, weight))
    result = to_sum[0]
    for result_piece in to_sum[1:]:
        result = result + result_piece
    return result


def _get_combination_and_multiply(combination: str,
                                  tensors: List[torch.Tensor],
                                  weight: torch.nn.Parameter) -> torch.Tensor:
    if combination.isdigit():
        index = int(combination) - 1
        return torch.matmul(tensors[index], weight)
    else:
        if len(combination) != 3:
            raise ConfigurationError("Invalid combination: " + combination)
        first_tensor = _get_combination(combination[0], tensors)
        second_tensor = _get_combination(combination[2], tensors)
        operation = combination[1]
        if operation == '*':
            if first_tensor.dim() > 4 or second_tensor.dim() > 4:
                raise ValueError("Tensors with dim > 4 not currently supported")
            desired_dim = max(first_tensor.dim(), second_tensor.dim()) - 1
            if first_tensor.dim() == 4:
                expanded_dim = _rindex(first_tensor.size(), 1)
                first_tensor = first_tensor.squeeze(expanded_dim)
            if second_tensor.dim() == 4:
                expanded_dim = _rindex(second_tensor.size(), 1)
                second_tensor = second_tensor.squeeze(expanded_dim)
            intermediate = first_tensor * weight
            result = torch.matmul(intermediate, second_tensor.transpose(-1, -2))
            if result.dim() == desired_dim + 1:
                result = result.squeeze(-1)
            return result
        elif operation == '/':
            if first_tensor.dim() > 4 or second_tensor.dim() > 4:
                raise ValueError("Tensors with dim > 4 not currently supported")
            desired_dim = max(first_tensor.dim(), second_tensor.dim()) - 1
            if first_tensor.dim() == 4:
                expanded_dim = _rindex(first_tensor.size(), 1)
                first_tensor = first_tensor.squeeze(expanded_dim)
            if second_tensor.dim() == 4:
                expanded_dim = _rindex(second_tensor.size(), 1)
                second_tensor = second_tensor.squeeze(expanded_dim)
            intermediate = first_tensor * weight
            result = torch.matmul(intermediate, second_tensor.pow(-1).transpose(-1, -2))
            if result.dim() == desired_dim + 1:
                result = result.squeeze(-1)
            return result
        elif operation == '+':
            return torch.matmul(first_tensor, weight) + torch.matmul(second_tensor, weight)
        elif operation == '-':
            return torch.matmul(first_tensor, weight) - torch.matmul(second_tensor, weight)
        else:
            raise ConfigurationError("Invalid operation: " + operation)


def get_combined_dim(combination: str, tensor_dims: List[int]) -> int:
    """
    For use with :func:`combine_tensors`.  This function computes the resultant dimension when
    calling ``combine_tensors(combination, tensors)``, when the tensor dimension is known.  This is
    necessary for knowing the sizes of weight matrices when building models that use
    ``combine_tensors``.

    Parameters
    ----------
    combination : ``str``
        A comma-separated list of combination pieces, like ``"1,2,1*2"``, specified identically to
        ``combination`` in :func:`combine_tensors`.
    tensor_dims : ``List[int]``
        A list of tensor dimensions, where each dimension is from the `last axis` of the tensors
        that will be input to :func:`combine_tensors`.
    """
    if len(tensor_dims) > 9:
        raise ConfigurationError("Double-digit tensor lists not currently supported")
    combination = combination.replace('x', '1').replace('y', '2')
    return sum([_get_combination_dim(piece, tensor_dims) for piece in combination.split(',')])


def _get_combination_dim(combination: str, tensor_dims: List[int]) -> int:
    if combination.isdigit():
        index = int(combination) - 1
        return tensor_dims[index]
    else:
        if len(combination) != 3:
            raise ConfigurationError("Invalid combination: " + combination)
        first_tensor_dim = _get_combination_dim(combination[0], tensor_dims)
        second_tensor_dim = _get_combination_dim(combination[2], tensor_dims)
        operation = combination[1]
        if first_tensor_dim != second_tensor_dim:
            raise ConfigurationError("Tensor dims must match for operation \"{}\"".format(operation))
        return first_tensor_dim


def logsumexp(tensor: torch.Tensor,
              dim: int = -1,
              keepdim: bool = False) -> torch.Tensor:
    """
    A numerically stable computation of logsumexp. This is mathematically equivalent to
    `tensor.exp().sum(dim, keep=keepdim).log()`.  This function is typically used for summing log
    probabilities.

    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A tensor of arbitrary size.
    dim : int, optional (default = -1)
        The dimension of the tensor to apply the logsumexp to.
    keepdim: bool, optional (default = False)
        Whether to retain a dimension of size one at the dimension we reduce over.
    """
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


def get_device_of(tensor: torch.Tensor) -> int:
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()


def flatten_and_batch_shift_indices(indices: torch.Tensor,
                                    sequence_length: int) -> torch.Tensor:
    """
    This is a subroutine for :func:`~batched_index_select`. The given ``indices`` of size
    ``(batch_size, d_1, ..., d_n)`` indexes into dimension 2 of a target tensor, which has size
    ``(batch_size, sequence_length, embedding_size)``. This function returns a vector that
    correctly indexes into the flattened target. The sequence length of the target must be
    provided to compute the appropriate offsets.

    .. code-block:: python

        indices = torch.ones([2,3], dtype=torch.long)
        # Sequence length of the target tensor.
        sequence_length = 10
        shifted_indices = flatten_and_batch_shift_indices(indices, sequence_length)
        # Indices into the second element in the batch are correctly shifted
        # to take into account that the target tensor will be flattened before
        # the indices are applied.
        assert shifted_indices == [1, 1, 1, 11, 11, 11]

    Parameters
    ----------
    indices : ``torch.LongTensor``, required.
    sequence_length : ``int``, required.
        The length of the sequence the indices index into.
        This must be the second dimension of the tensor.

    Returns
    -------
    offset_indices : ``torch.LongTensor``
    """
    # Shape: (batch_size)
    offsets = get_range_vector(indices.size(0), get_device_of(indices)) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices


def batched_index_select(target: torch.Tensor,
                         indices: torch.LongTensor,
                         flattened_indices: Optional[torch.LongTensor] = None) -> torch.Tensor:
    """
    The given ``indices`` of size ``(batch_size, d_1, ..., d_n)`` indexes into the sequence
    dimension (dimension 2) of the target, which has size ``(batch_size, sequence_length,
    embedding_size)``.

    This function returns selected values in the target with respect to the provided indices, which
    have size ``(batch_size, d_1, ..., d_n, embedding_size)``. This can use the optionally
    precomputed :func:`~flattened_indices` with size ``(batch_size * d_1 * ... * d_n)`` if given.

    An example use case of this function is looking up the start and end indices of spans in a
    sequence tensor. This is used in the
    :class:`~allennlp.models.coreference_resolution.CoreferenceResolver`. Model to select
    contextual word representations corresponding to the start and end indices of mentions. The key
    reason this can't be done with basic torch functions is that we want to be able to use look-up
    tensors with an arbitrary number of dimensions (for example, in the coref model, we don't know
    a-priori how many spans we are looking up).

    Parameters
    ----------
    target : ``torch.Tensor``, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : ``torch.LongTensor``
        A tensor of shape (batch_size, ...), where each element is an index into the
        ``sequence_length`` dimension of the ``target`` tensor.
    flattened_indices : Optional[torch.Tensor], optional (default = None)
        An optional tensor representing the result of calling :func:~`flatten_and_batch_shift_indices`
        on ``indices``. This is helpful in the case that the indices can be flattened once and
        cached for many batch lookups.

    Returns
    -------
    selected_targets : ``torch.Tensor``
        A tensor with shape [indices.size(), target.size(-1)] representing the embedded indices
        extracted from the batch flattened target tensor.
    """
    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))

    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets


def flattened_index_select(target: torch.Tensor,
                           indices: torch.LongTensor) -> torch.Tensor:
    """
    The given ``indices`` of size ``(set_size, subset_size)`` specifies subsets of the ``target``
    that each of the set_size rows should select. The `target` has size
    ``(batch_size, sequence_length, embedding_size)``, and the resulting selected tensor has size
    ``(batch_size, set_size, subset_size, embedding_size)``.

    Parameters
    ----------
    target : ``torch.Tensor``, required.
        A Tensor of shape (batch_size, sequence_length, embedding_size).
    indices : ``torch.LongTensor``, required.
        A LongTensor of shape (set_size, subset_size). All indices must be < sequence_length
        as this tensor is an index into the sequence_length dimension of the target.

    Returns
    -------
    selected : ``torch.Tensor``, required.
        A Tensor of shape (batch_size, set_size, subset_size, embedding_size).
    """
    if indices.dim() != 2:
        raise ConfigurationError("Indices passed to flattened_index_select had shape {} but "
                                 "only 2 dimensional inputs are supported.".format(indices.size()))
    # Shape: (batch_size, set_size * subset_size, embedding_size)
    flattened_selected = target.index_select(1, indices.view(-1))

    # Shape: (batch_size, set_size, subset_size, embedding_size)
    selected = flattened_selected.view(target.size(0), indices.size(0), indices.size(1), -1)
    return selected


def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)


def bucket_values(distances: torch.Tensor,
                  num_identity_buckets: int = 4,
                  num_total_buckets: int = 10) -> torch.Tensor:
    """
    Places the given values (designed for distances) into ``num_total_buckets``semi-logscale
    buckets, with ``num_identity_buckets`` of these capturing single values.

    The default settings will bucket values into the following buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].

    Parameters
    ----------
    distances : ``torch.Tensor``, required.
        A Tensor of any size, to be bucketed.
    num_identity_buckets: int, optional (default = 4).
        The number of identity buckets (those only holding a single value).
    num_total_buckets : int, (default = 10)
        The total number of buckets to bucket values into.

    Returns
    -------
    A tensor of the same shape as the input, containing the indices of the buckets
    the values were placed in.
    """
    # Chunk the values into semi-logscale buckets using .floor().
    # This is a semi-logscale bucketing because we divide by log(2) after taking the log.
    # We do this to make the buckets more granular in the initial range, where we expect
    # most values to fall. We then add (num_identity_buckets - 1) because we want these indices
    # to start _after_ the fixed number of buckets which we specified would only hold single values.
    logspace_index = (distances.float().log() / math.log(2)).floor().long() + (num_identity_buckets - 1)
    # create a mask for values which will go into single number buckets (i.e not a range).
    use_identity_mask = (distances <= num_identity_buckets).long()
    use_buckets_mask = 1 + (-1 * use_identity_mask)
    # Use the original values if they are less than num_identity_buckets, otherwise
    # use the logspace indices.
    combined_index = use_identity_mask * distances + use_buckets_mask * logspace_index
    # Clamp to put anything > num_total_buckets into the final bucket.
    return combined_index.clamp(0, num_total_buckets - 1)


def add_sentence_boundary_token_ids(tensor: torch.Tensor,
                                    mask: torch.Tensor,
                                    sentence_begin_token: Any,
                                    sentence_end_token: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add begin/end of sentence tokens to the batch of sentences.
    Given a batch of sentences with size ``(batch_size, timesteps)`` or
    ``(batch_size, timesteps, dim)`` this returns a tensor of shape
    ``(batch_size, timesteps + 2)`` or ``(batch_size, timesteps + 2, dim)`` respectively.

    Returns both the new tensor and updated mask.

    Parameters
    ----------
    tensor : ``torch.Tensor``
        A tensor of shape ``(batch_size, timesteps)`` or ``(batch_size, timesteps, dim)``
    mask : ``torch.Tensor``
         A tensor of shape ``(batch_size, timesteps)``
    sentence_begin_token: Any (anything that can be broadcast in torch for assignment)
        For 2D input, a scalar with the <S> id. For 3D input, a tensor with length dim.
    sentence_end_token: Any (anything that can be broadcast in torch for assignment)
        For 2D input, a scalar with the </S> id. For 3D input, a tensor with length dim.

    Returns
    -------
    tensor_with_boundary_tokens : ``torch.Tensor``
        The tensor with the appended and prepended boundary tokens. If the input was 2D,
        it has shape (batch_size, timesteps + 2) and if the input was 3D, it has shape
        (batch_size, timesteps + 2, dim).
    new_mask : ``torch.Tensor``
        The new mask for the tensor, taking into account the appended tokens
        marking the beginning and end of the sentence.
    """
    # TODO: matthewp, profile this transfer
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    tensor_shape = list(tensor.data.shape)
    new_shape = list(tensor_shape)
    new_shape[1] = tensor_shape[1] + 2
    tensor_with_boundary_tokens = tensor.new_zeros(*new_shape)
    if len(tensor_shape) == 2:
        tensor_with_boundary_tokens[:, 1:-1] = tensor
        tensor_with_boundary_tokens[:, 0] = sentence_begin_token
        for i, j in enumerate(sequence_lengths):
            tensor_with_boundary_tokens[i, j + 1] = sentence_end_token
        new_mask = (tensor_with_boundary_tokens != 0).long()
    elif len(tensor_shape) == 3:
        tensor_with_boundary_tokens[:, 1:-1, :] = tensor
        for i, j in enumerate(sequence_lengths):
            tensor_with_boundary_tokens[i, 0, :] = sentence_begin_token
            tensor_with_boundary_tokens[i, j + 1, :] = sentence_end_token
        new_mask = ((tensor_with_boundary_tokens > 0).long().sum(dim=-1) > 0).long()
    else:
        raise ValueError("add_sentence_boundary_token_ids only accepts 2D and 3D input")

    return tensor_with_boundary_tokens, new_mask


def remove_sentence_boundaries(tensor: torch.Tensor,
                               mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove begin/end of sentence embeddings from the batch of sentences.
    Given a batch of sentences with size ``(batch_size, timesteps, dim)``
    this returns a tensor of shape ``(batch_size, timesteps - 2, dim)`` after removing
    the beginning and end sentence markers.  The sentences are assumed to be padded on the right,
    with the beginning of each sentence assumed to occur at index 0 (i.e., ``mask[:, 0]`` is assumed
    to be 1).

    Returns both the new tensor and updated mask.

    This function is the inverse of ``add_sentence_boundary_token_ids``.

    Parameters
    ----------
    tensor : ``torch.Tensor``
        A tensor of shape ``(batch_size, timesteps, dim)``
    mask : ``torch.Tensor``
         A tensor of shape ``(batch_size, timesteps)``

    Returns
    -------
    tensor_without_boundary_tokens : ``torch.Tensor``
        The tensor after removing the boundary tokens of shape ``(batch_size, timesteps - 2, dim)``
    new_mask : ``torch.Tensor``
        The new mask for the tensor of shape ``(batch_size, timesteps - 2)``.
    """
    # TODO: matthewp, profile this transfer
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    tensor_shape = list(tensor.data.shape)
    new_shape = list(tensor_shape)
    new_shape[1] = tensor_shape[1] - 2
    tensor_without_boundary_tokens = tensor.new_zeros(*new_shape)
    new_mask = tensor.new_zeros((new_shape[0], new_shape[1]), dtype=torch.long)
    for i, j in enumerate(sequence_lengths):
        if j > 2:
            tensor_without_boundary_tokens[i, :(j - 2), :] = tensor[i, 1:(j - 1), :]
            new_mask[i, :(j - 2)] = 1

    return tensor_without_boundary_tokens, new_mask


def add_positional_features(tensor: torch.Tensor,
                            min_timescale: float = 1.0,
                            max_timescale: float = 1.0e4):
    # pylint: disable=line-too-long
    """
    Implements the frequency-based positional encoding described
    in `Attention is all you Need
    <https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077>`_ .

    Adds sinusoids of different frequencies to a ``Tensor``. A sinusoid of a
    different frequency and phase is added to each dimension of the input ``Tensor``.
    This allows the attention heads to use absolute and relative positions.

    The number of timescales is equal to hidden_dim / 2 within the range
    (min_timescale, max_timescale). For each timescale, the two sinusoidal
    signals sin(timestep / timescale) and cos(timestep / timescale) are
    generated and concatenated along the hidden_dim dimension.

    Parameters
    ----------
    tensor : ``torch.Tensor``
        a Tensor with shape (batch_size, timesteps, hidden_dim).
    min_timescale : ``float``, optional (default = 1.0)
        The smallest timescale to use.
    max_timescale : ``float``, optional (default = 1.0e4)
        The largest timescale to use.

    Returns
    -------
    The input tensor augmented with the sinusoidal frequencies.
    """
    _, timesteps, hidden_dim = tensor.size()

    timestep_range = get_range_vector(timesteps, get_device_of(tensor)).data.float()
    # We're generating both cos and sin frequencies,
    # so half for each.
    num_timescales = hidden_dim // 2
    timescale_range = get_range_vector(num_timescales, get_device_of(tensor)).data.float()

    log_timescale_increments = math.log(float(max_timescale) / float(min_timescale)) / float(num_timescales - 1)
    inverse_timescales = min_timescale * torch.exp(timescale_range * -log_timescale_increments)

    # Broadcasted multiplication - shape (timesteps, num_timescales)
    scaled_time = timestep_range.unsqueeze(1) * inverse_timescales.unsqueeze(0)
    # shape (timesteps, 2 * num_timescales)
    sinusoids = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 1)
    if hidden_dim % 2 != 0:
        # if the number of dimensions is odd, the cos and sin
        # timescales had size (hidden_dim - 1) / 2, so we need
        # to add a row of zeros to make up the difference.
        sinusoids = torch.cat([sinusoids, sinusoids.new_zeros(timesteps, 1)], 1)
    return tensor + sinusoids.unsqueeze(0)


def clone(module: torch.nn.Module, num_copies: int) -> torch.nn.ModuleList:
    """Produce N identical layers."""
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(num_copies)])


def combine_initial_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Given a (possibly higher order) tensor of ids with shape
    (d1, ..., dn, sequence_length)
    Return a view that's (d1 * ... * dn, sequence_length).
    If original tensor is 1-d or 2-d, return it as is.
    """
    if tensor.dim() <= 2:
        return tensor
    else:
        return tensor.view(-1, tensor.size(-1))


def uncombine_initial_dims(tensor: torch.Tensor, original_size: torch.Size) -> torch.Tensor:
    """
    Given a tensor of embeddings with shape
    (d1 * ... * dn, sequence_length, embedding_dim)
    and the original shape
    (d1, ..., dn, sequence_length),
    return the reshaped tensor of embeddings with shape
    (d1, ..., dn, sequence_length, embedding_dim).
    If original size is 1-d or 2-d, return it as is.
    """
    if len(original_size) <= 2:
        return tensor
    else:
        view_args = list(original_size) + [tensor.size(-1)]
        return tensor.view(*view_args)
