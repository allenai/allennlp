"""
Assorted utilities for working with neural networks in AllenNLP.
"""

import copy
from collections import defaultdict, OrderedDict
from itertools import chain
import json
import logging
from os import PathLike
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union, NamedTuple

import math
import numpy
import torch
import torch.distributed as dist

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import int_to_device, is_distributed, is_global_primary

logger = logging.getLogger(__name__)

T = TypeVar("T")
StateDictType = Union[Dict[str, torch.Tensor], "OrderedDict[str, torch.Tensor]"]

_MODULE_SHARDED_FLAG = "_is_sharded_allennlp"
"""
This flag is used to indicate when a module's parameters have been sharded across
distributed workers.
"""


def move_to_device(obj, device: Union[torch.device, int]):
    """
    Given a structure (possibly) containing Tensors,
    move all the Tensors to the specified device (or do nothing, if they are already on
    the target device).
    """
    device = int_to_device(device)

    if isinstance(obj, torch.Tensor):
        # You may be wondering why we don't just always call `obj.to(device)` since that would
        # be a no-op anyway if `obj` is already on `device`. Well that works fine except
        # when PyTorch is not compiled with CUDA support, in which case even calling
        # `obj.to(torch.device("cpu"))` would result in an error.
        return obj if obj.device == device else obj.to(device=device)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = move_to_device(value, device)
        return obj
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            obj[i] = move_to_device(item, device)
        return obj
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # This is the best way to detect a NamedTuple, it turns out.
        return obj.__class__(*(move_to_device(item, device) for item in obj))
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
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

        coalesced_tensor._values().clamp_(minimum, maximum)
        return coalesced_tensor
    else:
        return tensor.clamp(minimum, maximum)


def batch_tensor_dicts(
    tensor_dicts: List[Dict[str, torch.Tensor]], remove_trailing_dimension: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Takes a list of tensor dictionaries, where each dictionary is assumed to have matching keys,
    and returns a single dictionary with all tensors with the same key batched together.

    # Parameters

    tensor_dicts : `List[Dict[str, torch.Tensor]]`
        The list of tensor dictionaries to batch.
    remove_trailing_dimension : `bool`
        If `True`, we will check for a trailing dimension of size 1 on the tensors that are being
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


def get_lengths_from_binary_sequence_mask(mask: torch.BoolTensor) -> torch.LongTensor:
    """
    Compute sequence lengths for each batch element in a tensor using a
    binary mask.

    # Parameters

    mask : `torch.BoolTensor`, required.
        A 2D binary mask of shape (batch_size, sequence_length) to
        calculate the per-batch sequence lengths from.

    # Returns

    `torch.LongTensor`
        A torch.LongTensor of shape (batch_size,) representing the lengths
        of the sequences in the batch.
    """
    return mask.sum(-1)


def get_mask_from_sequence_lengths(
    sequence_lengths: torch.Tensor, max_length: int
) -> torch.BoolTensor:
    """
    Given a variable of shape `(batch_size,)` that represents the sequence lengths of each batch
    element, this function returns a `(batch_size, max_length)` mask variable.  For example, if
    our input was `[2, 2, 3]`, with a `max_length` of 4, we'd return
    `[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]`.

    We require `max_length` here instead of just computing it from the input `sequence_lengths`
    because it lets us avoid finding the max, then copying that value from the GPU to the CPU so
    that we can use it to construct a new tensor.
    """
    # (batch_size, max_length)
    ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return sequence_lengths.unsqueeze(1) >= range_tensor


def sort_batch_by_length(tensor: torch.Tensor, sequence_lengths: torch.Tensor):
    """
    Sort a batch first tensor by some specified lengths.

    # Parameters

    tensor : `torch.FloatTensor`, required.
        A batch first Pytorch tensor.
    sequence_lengths : `torch.LongTensor`, required.
        A tensor representing the lengths of some dimension of the tensor which
        we want to sort by.

    # Returns

    sorted_tensor : `torch.FloatTensor`
        The original tensor sorted along the batch dimension with respect to sequence_lengths.
    sorted_sequence_lengths : `torch.LongTensor`
        The original sequence_lengths sorted by decreasing size.
    restoration_indices : `torch.LongTensor`
        Indices into the sorted_tensor such that
        `sorted_tensor.index_select(0, restoration_indices) == original_tensor`
    permutation_index : `torch.LongTensor`
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


def get_final_encoder_states(
    encoder_outputs: torch.Tensor, mask: torch.BoolTensor, bidirectional: bool = False
) -> torch.Tensor:
    """
    Given the output from a `Seq2SeqEncoder`, with shape `(batch_size, sequence_length,
    encoding_dim)`, this method returns the final hidden state for each element of the batch,
    giving a tensor of shape `(batch_size, encoding_dim)`.  This is not as simple as
    `encoder_outputs[:, -1]`, because the sequences could have different lengths.  We use the
    mask (which has shape `(batch_size, sequence_length)`) to find the final state for each batch
    instance.

    Additionally, if `bidirectional` is `True`, we will split the final dimension of the
    `encoder_outputs` into two and assume that the first half is for the forward direction of the
    encoder and the second half is for the backward direction.  We will concatenate the last state
    for each encoder dimension, giving `encoder_outputs[:, -1, :encoding_dim/2]` concatenated with
    `encoder_outputs[:, 0, encoding_dim/2:]`.
    """
    # These are the indices of the last words in the sequences (i.e. length sans padding - 1).  We
    # are assuming sequences are right padded.
    # Shape: (batch_size,)
    last_word_indices = mask.sum(1) - 1
    batch_size, _, encoder_output_dim = encoder_outputs.size()
    expanded_indices = last_word_indices.view(-1, 1, 1).expand(batch_size, 1, encoder_output_dim)
    # Shape: (batch_size, 1, encoder_output_dim)
    final_encoder_output = encoder_outputs.gather(1, expanded_indices)
    final_encoder_output = final_encoder_output.squeeze(1)  # (batch_size, encoder_output_dim)
    if bidirectional:
        final_forward_output = final_encoder_output[:, : (encoder_output_dim // 2)]
        final_backward_output = encoder_outputs[:, 0, (encoder_output_dim // 2) :]
        final_encoder_output = torch.cat([final_forward_output, final_backward_output], dim=-1)
    return final_encoder_output


def get_dropout_mask(dropout_probability: float, tensor_for_masking: torch.Tensor):
    """
    Computes and returns an element-wise dropout mask for a given tensor, where
    each element in the mask is dropped out with probability dropout_probability.
    Note that the mask is NOT applied to the tensor - the tensor is passed to retain
    the correct CUDA tensor type for the mask.

    # Parameters

    dropout_probability : `float`, required.
        Probability of dropping a dimension of the input.
    tensor_for_masking : `torch.Tensor`, required.


    # Returns

    `torch.FloatTensor`
        A torch.FloatTensor consisting of the binary mask scaled by 1/ (1 - dropout_probability).
        This scaling ensures expected values and variances of the output of applying this mask
        and the original tensor are the same.
    """
    binary_mask = (torch.rand(tensor_for_masking.size()) > dropout_probability).to(
        tensor_for_masking.device
    )
    # Scale mask by 1/keep_prob to preserve output statistics.
    dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
    return dropout_mask


def masked_softmax(
    vector: torch.Tensor,
    mask: torch.BoolTensor,
    dim: int = -1,
    memory_efficient: bool = False,
) -> torch.Tensor:
    """
    `torch.nn.functional.softmax(vector)` does not work if some elements of `vector` should be
    masked.  This performs a softmax on just the non-masked portions of `vector`.  Passing
    `None` in for the mask is also acceptable; you'll just get a regular softmax.

    `vector` can have an arbitrary number of dimensions; the only requirement is that `mask` is
    broadcastable to `vector's` shape.  If `mask` has fewer dimensions than `vector`, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    If `memory_efficient` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.

    In the case that the input vector is completely masked and `memory_efficient` is false, this function
    returns an array of `0.0`. This behavior may cause `NaN` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if `memory_efficient` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (
                result.sum(dim=dim, keepdim=True) + tiny_value_of_dtype(result.dtype)
            )
        else:
            masked_vector = vector.masked_fill(~mask, min_value_of_dtype(vector.dtype))
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def masked_log_softmax(vector: torch.Tensor, mask: torch.BoolTensor, dim: int = -1) -> torch.Tensor:
    """
    `torch.nn.functional.log_softmax(vector)` does not work if some elements of `vector` should be
    masked.  This performs a log_softmax on just the non-masked portions of `vector`.  Passing
    `None` in for the mask is also acceptable; you'll just get a regular log_softmax.

    `vector` can have an arbitrary number of dimensions; the only requirement is that `mask` is
    broadcastable to `vector's` shape.  If `mask` has fewer dimensions than `vector`, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not `nan`.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you `nans`.

    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.
        vector = vector + (mask + tiny_value_of_dtype(vector.dtype)).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


def masked_max(
    vector: torch.Tensor,
    mask: torch.BoolTensor,
    dim: int,
    keepdim: bool = False,
) -> torch.Tensor:
    """
    To calculate max along certain dimensions on masked values

    # Parameters

    vector : `torch.Tensor`
        The vector to calculate max, assume unmasked parts are already zeros
    mask : `torch.BoolTensor`
        The mask of the vector. It must be broadcastable with vector.
    dim : `int`
        The dimension to calculate max
    keepdim : `bool`
        Whether to keep dimension

    # Returns

    `torch.Tensor`
        A `torch.Tensor` of including the maximum values.
    """
    replaced_vector = vector.masked_fill(~mask, min_value_of_dtype(vector.dtype))
    max_value, _ = replaced_vector.max(dim=dim, keepdim=keepdim)
    return max_value


def masked_mean(
    vector: torch.Tensor, mask: torch.BoolTensor, dim: int, keepdim: bool = False
) -> torch.Tensor:
    """
    To calculate mean along certain dimensions on masked values

    # Parameters

    vector : `torch.Tensor`
        The vector to calculate mean.
    mask : `torch.BoolTensor`
        The mask of the vector. It must be broadcastable with vector.
    dim : `int`
        The dimension to calculate mean
    keepdim : `bool`
        Whether to keep dimension

    # Returns

    `torch.Tensor`
        A `torch.Tensor` of including the mean values.
    """
    replaced_vector = vector.masked_fill(~mask, 0.0)

    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask, dim=dim, keepdim=keepdim)
    return value_sum / value_count.float().clamp(min=tiny_value_of_dtype(torch.float))


def masked_flip(padded_sequence: torch.Tensor, sequence_lengths: List[int]) -> torch.Tensor:
    """
    Flips a padded tensor along the time dimension without affecting masked entries.

    # Parameters

    padded_sequence : `torch.Tensor`
        The tensor to flip along the time dimension.
        Assumed to be of dimensions (batch size, num timesteps, ...)
    sequence_lengths : `torch.Tensor`
        A list containing the lengths of each unpadded sequence in the batch.

    # Returns

    `torch.Tensor`
        A `torch.Tensor` of the same shape as padded_sequence.
    """
    assert padded_sequence.size(0) == len(
        sequence_lengths
    ), f"sequence_lengths length ${len(sequence_lengths)} does not match batch size ${padded_sequence.size(0)}"
    num_timesteps = padded_sequence.size(1)
    flipped_padded_sequence = torch.flip(padded_sequence, [1])
    sequences = [
        flipped_padded_sequence[i, num_timesteps - length :]
        for i, length in enumerate(sequence_lengths)
    ]
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)


def viterbi_decode(
    tag_sequence: torch.Tensor,
    transition_matrix: torch.Tensor,
    tag_observations: Optional[List[int]] = None,
    allowed_start_transitions: torch.Tensor = None,
    allowed_end_transitions: torch.Tensor = None,
    top_k: int = None,
):
    """
    Perform Viterbi decoding in log space over a sequence given a transition matrix
    specifying pairwise (transition) potentials between tags and a matrix of shape
    (sequence_length, num_tags) specifying unary potentials for possible tags per
    timestep.

    # Parameters

    tag_sequence : `torch.Tensor`, required.
        A tensor of shape (sequence_length, num_tags) representing scores for
        a set of tags over a given sequence.
    transition_matrix : `torch.Tensor`, required.
        A tensor of shape (num_tags, num_tags) representing the binary potentials
        for transitioning between a given pair of tags.
    tag_observations : `Optional[List[int]]`, optional, (default = `None`)
        A list of length `sequence_length` containing the class ids of observed
        elements in the sequence, with unobserved elements being set to -1. Note that
        it is possible to provide evidence which results in degenerate labelings if
        the sequences of tags you provide as evidence cannot transition between each
        other, or those transitions are extremely unlikely. In this situation we log a
        warning, but the responsibility for providing self-consistent evidence ultimately
        lies with the user.
    allowed_start_transitions : `torch.Tensor`, optional, (default = `None`)
        An optional tensor of shape (num_tags,) describing which tags the START token
        may transition *to*. If provided, additional transition constraints will be used for
        determining the start element of the sequence.
    allowed_end_transitions : `torch.Tensor`, optional, (default = `None`)
        An optional tensor of shape (num_tags,) describing which tags may transition *to* the
        end tag. If provided, additional transition constraints will be used for determining
        the end element of the sequence.
    top_k : `int`, optional, (default = `None`)
        Optional integer specifying how many of the top paths to return. For top_k>=1, returns
        a tuple of two lists: top_k_paths, top_k_scores, For top_k==None, returns a flattened
        tuple with just the top path and its score (not in lists, for backwards compatibility).

    # Returns

    viterbi_path : `List[int]`
        The tag indices of the maximum likelihood tag sequence.
    viterbi_score : `torch.Tensor`
        The score of the viterbi path.
    """
    if top_k is None:
        top_k = 1
        flatten_output = True
    elif top_k >= 1:
        flatten_output = False
    else:
        raise ValueError(f"top_k must be either None or an integer >=1. Instead received {top_k}")

    sequence_length, num_tags = list(tag_sequence.size())

    has_start_end_restrictions = (
        allowed_end_transitions is not None or allowed_start_transitions is not None
    )

    if has_start_end_restrictions:

        if allowed_end_transitions is None:
            allowed_end_transitions = torch.zeros(num_tags)
        if allowed_start_transitions is None:
            allowed_start_transitions = torch.zeros(num_tags)

        num_tags = num_tags + 2
        new_transition_matrix = torch.zeros(num_tags, num_tags)
        new_transition_matrix[:-2, :-2] = transition_matrix

        # Start and end transitions are fully defined, but cannot transition between each other.

        allowed_start_transitions = torch.cat(
            [allowed_start_transitions, torch.tensor([-math.inf, -math.inf])]
        )
        allowed_end_transitions = torch.cat(
            [allowed_end_transitions, torch.tensor([-math.inf, -math.inf])]
        )

        # First define how we may transition FROM the start and end tags.
        new_transition_matrix[-2, :] = allowed_start_transitions
        # We cannot transition from the end tag to any tag.
        new_transition_matrix[-1, :] = -math.inf

        new_transition_matrix[:, -1] = allowed_end_transitions
        # We cannot transition to the start tag from any tag.
        new_transition_matrix[:, -2] = -math.inf

        transition_matrix = new_transition_matrix

    if tag_observations:
        if len(tag_observations) != sequence_length:
            raise ConfigurationError(
                "Observations were provided, but they were not the same length "
                "as the sequence. Found sequence of length: {} and evidence: {}".format(
                    sequence_length, tag_observations
                )
            )
    else:
        tag_observations = [-1 for _ in range(sequence_length)]

    if has_start_end_restrictions:
        tag_observations = [num_tags - 2] + tag_observations + [num_tags - 1]
        zero_sentinel = torch.zeros(1, num_tags)
        extra_tags_sentinel = torch.ones(sequence_length, 2) * -math.inf
        tag_sequence = torch.cat([tag_sequence, extra_tags_sentinel], -1)
        tag_sequence = torch.cat([zero_sentinel, tag_sequence, zero_sentinel], 0)
        sequence_length = tag_sequence.size(0)

    path_scores = []
    path_indices = []

    if tag_observations[0] != -1:
        one_hot = torch.zeros(num_tags)
        one_hot[tag_observations[0]] = 100000.0
        path_scores.append(one_hot.unsqueeze(0))
    else:
        path_scores.append(tag_sequence[0, :].unsqueeze(0))

    # Evaluate the scores for all possible paths.
    for timestep in range(1, sequence_length):
        # Add pairwise potentials to current scores.
        summed_potentials = path_scores[timestep - 1].unsqueeze(2) + transition_matrix
        summed_potentials = summed_potentials.view(-1, num_tags)

        # Best pairwise potential path score from the previous timestep.
        max_k = min(summed_potentials.size()[0], top_k)
        scores, paths = torch.topk(summed_potentials, k=max_k, dim=0)

        # If we have an observation for this timestep, use it
        # instead of the distribution over tags.
        observation = tag_observations[timestep]
        # Warn the user if they have passed
        # invalid/extremely unlikely evidence.
        if tag_observations[timestep - 1] != -1 and observation != -1:
            if transition_matrix[tag_observations[timestep - 1], observation] < -10000:
                logger.warning(
                    "The pairwise potential between tags you have passed as "
                    "observations is extremely unlikely. Double check your evidence "
                    "or transition potentials!"
                )
        if observation != -1:
            one_hot = torch.zeros(num_tags)
            one_hot[observation] = 100000.0
            path_scores.append(one_hot.unsqueeze(0))
        else:
            path_scores.append(tag_sequence[timestep, :] + scores)
        path_indices.append(paths.squeeze())

    # Construct the most likely sequence backwards.
    path_scores_v = path_scores[-1].view(-1)
    max_k = min(path_scores_v.size()[0], top_k)
    viterbi_scores, best_paths = torch.topk(path_scores_v, k=max_k, dim=0)
    viterbi_paths = []
    for i in range(max_k):
        viterbi_path = [best_paths[i]]
        for backward_timestep in reversed(path_indices):
            viterbi_path.append(int(backward_timestep.view(-1)[viterbi_path[-1]]))
        # Reverse the backward path.
        viterbi_path.reverse()

        if has_start_end_restrictions:
            viterbi_path = viterbi_path[1:-1]

        # Viterbi paths uses (num_tags * n_permutations) nodes; therefore, we need to modulo.
        viterbi_path = [j % num_tags for j in viterbi_path]
        viterbi_paths.append(viterbi_path)

    if flatten_output:
        return viterbi_paths[0], viterbi_scores[0]

    return viterbi_paths, viterbi_scores


def get_text_field_mask(
    text_field_tensors: Dict[str, Dict[str, torch.Tensor]],
    num_wrapping_dims: int = 0,
    padding_id: int = 0,
) -> torch.BoolTensor:
    """
    Takes the dictionary of tensors produced by a `TextField` and returns a mask
    with 0 where the tokens are padding, and 1 otherwise. `padding_id` specifies the id of padding tokens.
    We also handle `TextFields` wrapped by an arbitrary number of `ListFields`, where the number of wrapping
    `ListFields` is given by `num_wrapping_dims`.

    If `num_wrapping_dims == 0`, the returned mask has shape `(batch_size, num_tokens)`.
    If `num_wrapping_dims > 0` then the returned mask has `num_wrapping_dims` extra
    dimensions, so the shape will be `(batch_size, ..., num_tokens)`.

    There could be several entries in the tensor dictionary with different shapes (e.g., one for
    word ids, one for character ids).  In order to get a token mask, we use the tensor in
    the dictionary with the lowest number of dimensions.  After subtracting `num_wrapping_dims`,
    if this tensor has two dimensions we assume it has shape `(batch_size, ..., num_tokens)`,
    and use it for the mask.  If instead it has three dimensions, we assume it has shape
    `(batch_size, ..., num_tokens, num_features)`, and sum over the last dimension to produce
    the mask.  Most frequently this will be a character id tensor, but it could also be a
    featurized representation of each token, etc.

    If the input `text_field_tensors` contains the "mask" key, this is returned instead of inferring the mask.
    """
    masks = []
    for indexer_name, indexer_tensors in text_field_tensors.items():
        if "mask" in indexer_tensors:
            masks.append(indexer_tensors["mask"].bool())
    if len(masks) == 1:
        return masks[0]
    elif len(masks) > 1:
        # TODO(mattg): My guess is this will basically never happen, so I'm not writing logic to
        # handle it.  Should be straightforward to handle, though.  If you see this error in
        # practice, open an issue on github.
        raise ValueError("found two mask outputs; not sure which to use!")

    tensor_dims = [
        (tensor.dim(), tensor)
        for indexer_output in text_field_tensors.values()
        for tensor in indexer_output.values()
    ]
    tensor_dims.sort(key=lambda x: x[0])

    smallest_dim = tensor_dims[0][0] - num_wrapping_dims
    if smallest_dim == 2:
        token_tensor = tensor_dims[0][1]
        return token_tensor != padding_id
    elif smallest_dim == 3:
        character_tensor = tensor_dims[0][1]
        return (character_tensor != padding_id).any(dim=-1)
    else:
        raise ValueError("Expected a tensor with dimension 2 or 3, found {}".format(smallest_dim))


def get_token_ids_from_text_field_tensors(
    text_field_tensors: Dict[str, Dict[str, torch.Tensor]],
) -> torch.Tensor:
    """
    Our `TextFieldTensors` are complex output structures, because they try to handle a lot of
    potential variation. Sometimes, you just want to grab the token ids from this data structure,
    and that's not trivial without hard-coding assumptions about your data processing, which defeats
    the entire purpose of that generality. This method tries to let you get the token ids out of the
    data structure in your model without hard-coding any assumptions.
    """
    for indexer_name, indexer_tensors in text_field_tensors.items():
        for argument_name, tensor in indexer_tensors.items():
            if argument_name in ["tokens", "token_ids", "input_ids"]:
                return tensor
    raise NotImplementedError(
        "Our heuristic for guessing the right token ids failed. Please open an issue on "
        "github with more detail on how you got this error, so we can implement more robust "
        "logic in this method."
    )


def weighted_sum(matrix: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    """
    Takes a matrix of vectors and a set of weights over the rows in the matrix (which we call an
    "attention" vector), and returns a weighted sum of the rows in the matrix.  This is the typical
    computation performed after an attention mechanism.

    Note that while we call this a "matrix" of vectors and an attention "vector", we also handle
    higher-order tensors.  We always sum over the second-to-last dimension of the "matrix", and we
    assume that all dimensions in the "matrix" prior to the last dimension are matched in the
    "vector".  Non-matched dimensions in the "vector" must be `directly after the batch dimension`.

    For example, say I have a "matrix" with dimensions `(batch_size, num_queries, num_words,
    embedding_dim)`.  The attention "vector" then must have at least those dimensions, and could
    have more. Both:

        - `(batch_size, num_queries, num_words)` (distribution over words for each query)
        - `(batch_size, num_documents, num_queries, num_words)` (distribution over words in a
          query for each document)

    are valid input "vectors", producing tensors of shape:
    `(batch_size, num_queries, embedding_dim)` and
    `(batch_size, num_documents, num_queries, embedding_dim)` respectively.
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


def sequence_cross_entropy_with_logits(
    logits: torch.FloatTensor,
    targets: torch.LongTensor,
    weights: Union[torch.FloatTensor, torch.BoolTensor],
    average: str = "batch",
    label_smoothing: float = None,
    gamma: float = None,
    alpha: Union[float, List[float], torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Computes the cross entropy loss of a sequence, weighted with respect to
    some user provided weights. Note that the weighting here is not the same as
    in the `torch.nn.CrossEntropyLoss()` criterion, which is weighting
    classes; here we are weighting the loss contribution from particular elements
    in the sequence. This allows loss computations for models which use padding.

    # Parameters

    logits : `torch.FloatTensor`, required.
        A `torch.FloatTensor` of size (batch_size, sequence_length, num_classes)
        which contains the unnormalized probability for each class.
    targets : `torch.LongTensor`, required.
        A `torch.LongTensor` of size (batch, sequence_length) which contains the
        index of the true class for each corresponding step.
    weights : `Union[torch.FloatTensor, torch.BoolTensor]`, required.
        A `torch.FloatTensor` of size (batch, sequence_length)
    average: `str`, optional (default = `"batch"`)
        If "batch", average the loss across the batches. If "token", average
        the loss across each item in the input. If `None`, return a vector
        of losses per batch element.
    label_smoothing : `float`, optional (default = `None`)
        Whether or not to apply label smoothing to the cross-entropy loss.
        For example, with a label smoothing value of 0.2, a 4 class classification
        target would look like `[0.05, 0.05, 0.85, 0.05]` if the 3rd class was
        the correct label.
    gamma : `float`, optional (default = `None`)
        Focal loss[*] focusing parameter `gamma` to reduces the relative loss for
        well-classified examples and put more focus on hard. The greater value
        `gamma` is, the more focus on hard examples.
    alpha : `Union[float, List[float]]`, optional (default = `None`)
        Focal loss[*] weighting factor `alpha` to balance between classes. Can be
        used independently with `gamma`. If a single `float` is provided, it
        is assumed binary case using `alpha` and `1 - alpha` for positive and
        negative respectively. If a list of `float` is provided, with the same
        length as the number of classes, the weights will match the classes.
        [*] T. Lin, P. Goyal, R. Girshick, K. He and P. Dollár, "Focal Loss for
        Dense Object Detection," 2017 IEEE International Conference on Computer
        Vision (ICCV), Venice, 2017, pp. 2999-3007.

    # Returns

    `torch.FloatTensor`
        A torch.FloatTensor representing the cross entropy loss.
        If `average=="batch"` or `average=="token"`, the returned loss is a scalar.
        If `average is None`, the returned loss is a vector of shape (batch_size,).

    """
    if average not in {None, "token", "batch"}:
        raise ValueError("Got average f{average}, expected one of None, 'token', or 'batch'")

    # make sure weights are float
    weights = weights.to(logits.dtype)
    # sum all dim except batch
    non_batch_dims = tuple(range(1, len(weights.shape)))
    # shape : (batch_size,)
    weights_batch_sum = weights.sum(dim=non_batch_dims)
    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()
    # focal loss coefficient
    if gamma:
        # shape : (batch * sequence_length, num_classes)
        probs_flat = log_probs_flat.exp()
        # shape : (batch * sequence_length,)
        probs_flat = torch.gather(probs_flat, dim=1, index=targets_flat)
        # shape : (batch * sequence_length,)
        focal_factor = (1.0 - probs_flat) ** gamma
        # shape : (batch, sequence_length)
        focal_factor = focal_factor.view(*targets.size())
        weights = weights * focal_factor

    if alpha is not None:
        # shape : () / (num_classes,)
        if isinstance(alpha, (float, int)):

            # shape : (2,)
            alpha_factor = torch.tensor(
                [1.0 - float(alpha), float(alpha)], dtype=weights.dtype, device=weights.device
            )

        elif isinstance(alpha, (list, numpy.ndarray, torch.Tensor)):

            # shape : (c,)
            alpha_factor = torch.tensor(alpha, dtype=weights.dtype, device=weights.device)

            if not alpha_factor.size():
                # shape : (1,)
                alpha_factor = alpha_factor.view(1)
                # shape : (2,)
                alpha_factor = torch.cat([1 - alpha_factor, alpha_factor])
        else:
            raise TypeError(
                ("alpha must be float, list of float, or torch.FloatTensor, {} provided.").format(
                    type(alpha)
                )
            )
        # shape : (batch, max_len)
        alpha_factor = torch.gather(alpha_factor, dim=0, index=targets_flat.view(-1)).view(
            *targets.size()
        )
        weights = weights * alpha_factor

    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / num_classes
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(
            -1, targets_flat, 1.0 - label_smoothing
        )
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = -log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # Contribution to the negative log likelihood only comes from the exact indices
        # of the targets, as the target distributions are one-hot. Here we use torch.gather
        # to extract the indices of the num_classes dimension which contribute to the loss.
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = -torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights

    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (
            weights_batch_sum + tiny_value_of_dtype(negative_log_likelihood.dtype)
        )
        num_non_empty_sequences = (weights_batch_sum > 0).sum() + tiny_value_of_dtype(
            negative_log_likelihood.dtype
        )
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (
            weights_batch_sum.sum() + tiny_value_of_dtype(negative_log_likelihood.dtype)
        )
    else:
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (
            weights_batch_sum + tiny_value_of_dtype(negative_log_likelihood.dtype)
        )
        return per_batch_loss


def replace_masked_values(
    tensor: torch.Tensor, mask: torch.BoolTensor, replace_with: float
) -> torch.Tensor:
    """
    Replaces all masked values in `tensor` with `replace_with`.  `mask` must be broadcastable
    to the same shape as `tensor`. We require that `tensor.dim() == mask.dim()`, as otherwise we
    won't know which dimensions of the mask to unsqueeze.

    This just does `tensor.masked_fill()`, except the pytorch method fills in things with a mask
    value of 1, where we want the opposite.  You can do this in your own code with
    `tensor.masked_fill(~mask, replace_with)`.
    """
    if tensor.dim() != mask.dim():
        raise ConfigurationError(
            "tensor.dim() (%d) != mask.dim() (%d)" % (tensor.dim(), mask.dim())
        )
    return tensor.masked_fill(~mask, replace_with)


def tensors_equal(tensor1: torch.Tensor, tensor2: torch.Tensor, tolerance: float = 1e-12) -> bool:
    """
    A check for tensor equality (by value).  We make sure that the tensors have the same shape,
    then check all of the entries in the tensor for equality.  We additionally allow the input
    tensors to be lists or dictionaries, where we then do the above check on every position in the
    list / item in the dictionary.  If we find objects that aren't tensors as we're doing that, we
    just defer to their equality check.

    This is kind of a catch-all method that's designed to make implementing `__eq__` methods
    easier, in a way that's really only intended to be useful for tests.
    """

    if isinstance(tensor1, (list, tuple)):
        if not isinstance(tensor2, (list, tuple)) or len(tensor1) != len(tensor2):
            return False
        return all(tensors_equal(t1, t2, tolerance) for t1, t2 in zip(tensor1, tensor2))
    elif isinstance(tensor1, dict):
        if not isinstance(tensor2, dict):
            return False
        if tensor1.keys() != tensor2.keys():
            return False
        return all(tensors_equal(tensor1[key], tensor2[key], tolerance) for key in tensor1)
    elif isinstance(tensor1, torch.Tensor):
        if not isinstance(tensor2, torch.Tensor):
            return False
        if tensor1.size() != tensor2.size():
            return False
        # Special case for bools since they don't support subtraction
        if tensor1.dtype == torch.bool or tensor2.dtype == torch.bool:
            return (tensor1 == tensor2).all()
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

    def inner_device_mapping(storage: torch.Storage, location) -> torch.Storage:
        if cuda_device >= 0:
            return storage.cuda(cuda_device)
        else:
            return storage

    return inner_device_mapping


def read_state_dict(
    path: Union[PathLike, str],
    strip_prefix: Optional[str] = None,
    ignore: Optional[List[str]] = None,
    strict: bool = True,
    cuda_device: int = -1,
) -> Dict[str, torch.Tensor]:
    """
    Read a PyTorch model state dictionary from a checkpoint at the given `path`.

    # Parameters

    path : `Union[PathLike, str]`, required

    strip_prefix : `Optional[str]`, optional (default = `None`)
        A prefix to remove from all of the state dict keys.

    ignore : `Optional[List[str]]`, optional (default = `None`)
        Optional list of regular expressions. Keys that match any of these will be removed
        from the state dict.

        !!! Note
            If `strip_prefix` is given, the regular expressions in `ignore` are matched
            before the prefix is stripped.

    strict : `bool`, optional (default = `True`)
        If `True` (the default) and `strip_prefix` was never used or any of the regular expressions
        in `ignore` never matched, a `ValueError` will be raised.

    cuda_device : `int`, optional (default = `-1`)
        The device to load the parameters onto. Use `-1` (the default) for CPU.

    # Returns

    `Dict[str, torch.Tensor]`
        An ordered dictionary of the state.
    """
    state = torch.load(path, map_location=device_mapping(cuda_device))
    out: Dict[str, torch.Tensor] = OrderedDict()

    if ignore is not None and not isinstance(ignore, list):
        # If user accidentally passed in something that is not a list - like a string,
        # which is easy to do - the user would be confused why the resulting state dict
        # is empty.
        raise ValueError("'ignore' parameter should be a list")

    # In 'strict' mode, we need to keep track of whether we've used `strip_prefix`
    # and which regular expressions in `ignore` we've used.
    strip_prefix_used: Optional[bool] = None
    ignore_used: Optional[List[bool]] = None
    if strict and strip_prefix is not None:
        strip_prefix_used = False
    if strict and ignore:
        ignore_used = [False] * len(ignore)

    for key in state.keys():
        ignore_key = False
        if ignore:
            for i, pattern in enumerate(ignore):
                if re.match(pattern, key):
                    if ignore_used:
                        ignore_used[i] = True
                    logger.warning("ignoring %s from state dict", key)
                    ignore_key = True
                    break

        if ignore_key:
            continue

        new_key = key

        if strip_prefix and key.startswith(strip_prefix):
            strip_prefix_used = True
            new_key = key[len(strip_prefix) :]
            if not new_key:
                raise ValueError("'strip_prefix' resulted in an empty string for a key")

        out[new_key] = state[key]

    if strip_prefix_used is False:
        raise ValueError(f"'strip_prefix' of '{strip_prefix}' was never used")
    if ignore is not None and ignore_used is not None:
        for pattern, used in zip(ignore, ignore_used):
            if not used:
                raise ValueError(f"'ignore' pattern '{pattern}' didn't have any matches")

    return out


def combine_tensors(combination: str, tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Combines a list of tensors using element-wise operations and concatenation, specified by a
    `combination` string.  The string refers to (1-indexed) positions in the input tensor list,
    and looks like `"1,2,1+2,3-1"`.

    We allow the following kinds of combinations : `x`, `x*y`, `x+y`, `x-y`, and `x/y`,
    where `x` and `y` are positive integers less than or equal to `len(tensors)`.  Each of
    the binary operations is performed elementwise.  You can give as many combinations as you want
    in the `combination` string.  For example, for the input string `"1,2,1*2"`, the result
    would be `[1;2;1*2]`, as you would expect, where `[;]` is concatenation along the last
    dimension.

    If you have a fixed, known way to combine tensors that you use in a model, you should probably
    just use something like `torch.cat([x_tensor, y_tensor, x_tensor * y_tensor])`.  This
    function adds some complexity that is only necessary if you want the specific combination used
    to be `configurable`.

    If you want to do any element-wise operations, the tensors involved in each element-wise
    operation must have the same shape.

    This function also accepts `x` and `y` in place of `1` and `2` in the combination
    string.
    """
    if len(tensors) > 9:
        raise ConfigurationError("Double-digit tensor lists not currently supported")
    combination = combination.replace("x", "1").replace("y", "2")
    to_concatenate = [_get_combination(piece, tensors) for piece in combination.split(",")]
    return torch.cat(to_concatenate, dim=-1)


def _rindex(sequence: Sequence[T], obj: T) -> int:
    """
    Return zero-based index in the sequence of the last item whose value is equal to obj.  Raises a
    ValueError if there is no such item.

    # Parameters

    sequence : `Sequence[T]`
    obj : `T`

    # Returns

    `int`
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
        if operation == "*":
            return first_tensor * second_tensor
        elif operation == "/":
            return first_tensor / second_tensor
        elif operation == "+":
            return first_tensor + second_tensor
        elif operation == "-":
            return first_tensor - second_tensor
        else:
            raise ConfigurationError("Invalid operation: " + operation)


def combine_tensors_and_multiply(
    combination: str, tensors: List[torch.Tensor], weights: torch.nn.Parameter
) -> torch.Tensor:
    """
    Like [`combine_tensors`](./util.md#combine_tensors), but does a weighted (linear)
    multiplication while combining. This is a separate function from `combine_tensors`
    because we try to avoid instantiating large intermediate tensors during the combination,
    which is possible because we know that we're going to be multiplying by a weight vector in the end.

    # Parameters

    combination : `str`
        Same as in `combine_tensors`
    tensors : `List[torch.Tensor]`
        A list of tensors to combine, where the integers in the `combination` are (1-indexed)
        positions in this list of tensors.  These tensors are all expected to have either three or
        four dimensions, with the final dimension being an embedding.  If there are four
        dimensions, one of them must have length 1.
    weights : `torch.nn.Parameter`
        A vector of weights to use for the combinations.  This should have shape (combined_dim,),
        as calculated by `get_combined_dim`.
    """
    if len(tensors) > 9:
        raise ConfigurationError("Double-digit tensor lists not currently supported")
    combination = combination.replace("x", "1").replace("y", "2")
    pieces = combination.split(",")
    tensor_dims = [tensor.size(-1) for tensor in tensors]
    combination_dims = [_get_combination_dim(piece, tensor_dims) for piece in pieces]
    dims_so_far = 0
    to_sum = []
    for piece, combination_dim in zip(pieces, combination_dims):
        weight = weights[dims_so_far : (dims_so_far + combination_dim)]
        dims_so_far += combination_dim
        to_sum.append(_get_combination_and_multiply(piece, tensors, weight))
    result = to_sum[0]
    for result_piece in to_sum[1:]:
        result = result + result_piece
    return result


def _get_combination_and_multiply(
    combination: str, tensors: List[torch.Tensor], weight: torch.nn.Parameter
) -> torch.Tensor:
    if combination.isdigit():
        index = int(combination) - 1
        return torch.matmul(tensors[index], weight)
    else:
        if len(combination) != 3:
            raise ConfigurationError("Invalid combination: " + combination)
        first_tensor = _get_combination(combination[0], tensors)
        second_tensor = _get_combination(combination[2], tensors)
        operation = combination[1]
        if operation == "*":
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
        elif operation == "/":
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
        elif operation == "+":
            return torch.matmul(first_tensor, weight) + torch.matmul(second_tensor, weight)
        elif operation == "-":
            return torch.matmul(first_tensor, weight) - torch.matmul(second_tensor, weight)
        else:
            raise ConfigurationError("Invalid operation: " + operation)


def get_combined_dim(combination: str, tensor_dims: List[int]) -> int:
    """
    For use with [`combine_tensors`](./util.md#combine_tensors).
    This function computes the resultant dimension when calling `combine_tensors(combination, tensors)`,
    when the tensor dimension is known.  This is necessary for knowing the sizes of weight matrices
    when building models that use `combine_tensors`.

    # Parameters

    combination : `str`
        A comma-separated list of combination pieces, like `"1,2,1*2"`, specified identically to
        `combination` in `combine_tensors`.
    tensor_dims : `List[int]`
        A list of tensor dimensions, where each dimension is from the `last axis` of the tensors
        that will be input to `combine_tensors`.
    """
    if len(tensor_dims) > 9:
        raise ConfigurationError("Double-digit tensor lists not currently supported")
    combination = combination.replace("x", "1").replace("y", "2")
    return sum(_get_combination_dim(piece, tensor_dims) for piece in combination.split(","))


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
            raise ConfigurationError('Tensor dims must match for operation "{}"'.format(operation))
        return first_tensor_dim


def logsumexp(tensor: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """
    A numerically stable computation of logsumexp. This is mathematically equivalent to
    `tensor.exp().sum(dim, keep=keepdim).log()`.  This function is typically used for summing log
    probabilities.

    # Parameters

    tensor : `torch.FloatTensor`, required.
        A tensor of arbitrary size.
    dim : `int`, optional (default = `-1`)
        The dimension of the tensor to apply the logsumexp to.
    keepdim: `bool`, optional (default = `False`)
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


def flatten_and_batch_shift_indices(indices: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """
    This is a subroutine for [`batched_index_select`](./util.md#batched_index_select).
    The given `indices` of size `(batch_size, d_1, ..., d_n)` indexes into dimension 2 of a
    target tensor, which has size `(batch_size, sequence_length, embedding_size)`. This
    function returns a vector that correctly indexes into the flattened target. The sequence
    length of the target must be provided to compute the appropriate offsets.

    ```python
        indices = torch.ones([2,3], dtype=torch.long)
        # Sequence length of the target tensor.
        sequence_length = 10
        shifted_indices = flatten_and_batch_shift_indices(indices, sequence_length)
        # Indices into the second element in the batch are correctly shifted
        # to take into account that the target tensor will be flattened before
        # the indices are applied.
        assert shifted_indices == [1, 1, 1, 11, 11, 11]
    ```

    # Parameters

    indices : `torch.LongTensor`, required.
    sequence_length : `int`, required.
        The length of the sequence the indices index into.
        This must be the second dimension of the tensor.

    # Returns

    offset_indices : `torch.LongTensor`
    """
    # Shape: (batch_size)
    if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
        raise ConfigurationError(
            f"All elements in indices should be in range (0, {sequence_length - 1})"
        )
    offsets = get_range_vector(indices.size(0), get_device_of(indices)) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices


def batched_index_select(
    target: torch.Tensor,
    indices: torch.LongTensor,
    flattened_indices: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    """
    The given `indices` of size `(batch_size, d_1, ..., d_n)` indexes into the sequence
    dimension (dimension 2) of the target, which has size `(batch_size, sequence_length,
    embedding_size)`.

    This function returns selected values in the target with respect to the provided indices, which
    have size `(batch_size, d_1, ..., d_n, embedding_size)`. This can use the optionally
    precomputed `flattened_indices` with size `(batch_size * d_1 * ... * d_n)` if given.

    An example use case of this function is looking up the start and end indices of spans in a
    sequence tensor. This is used in the
    [CoreferenceResolver](https://docs.allennlp.org/models/main/models/coref/models/coref/)
    model to select contextual word representations corresponding to the start and end indices of
    mentions.

    The key reason this can't be done with basic torch functions is that we want to be able to use look-up
    tensors with an arbitrary number of dimensions (for example, in the coref model, we don't know
    a-priori how many spans we are looking up).

    # Parameters

    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : `torch.LongTensor`
        A tensor of shape (batch_size, ...), where each element is an index into the
        `sequence_length` dimension of the `target` tensor.
    flattened_indices : `Optional[torch.Tensor]`, optional (default = `None`)
        An optional tensor representing the result of calling `flatten_and_batch_shift_indices`
        on `indices`. This is helpful in the case that the indices can be flattened once and
        cached for many batch lookups.

    # Returns

    selected_targets : `torch.Tensor`
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


def masked_index_fill(
    target: torch.Tensor, indices: torch.LongTensor, mask: torch.BoolTensor, fill_value: int = 1
) -> torch.Tensor:
    """
    The given `indices` in `target` will be will be filled with `fill_value` given a `mask`.


    # Parameters

    target : `torch.Tensor`, required.
        A 2 dimensional tensor of shape (batch_size, sequence_length).
        This is the tensor to be filled.
    indices : `torch.LongTensor`, required
        A 2 dimensional tensor of shape (batch_size, num_indices),
        These are the indices that will be filled in the original tensor.
    mask : `torch.Tensor`, required.
        A 2 dimensional tensor of shape (batch_size, num_indices), mask.sum() == `nonzero_indices`.
    fill_value : `int`, optional (default = `1`)
        The value we fill the tensor with.

    # Returns

    filled_target : `torch.Tensor`
        A tensor with shape (batch_size, sequence_length) where 'indices' are filled with `fill_value`
    """
    mask = mask.bool()
    prev_shape = target.size()
    # Shape: (batch_size * num_indices)
    flattened_indices = flatten_and_batch_shift_indices(indices * mask, target.size(1))
    # Shape: (batch_size * num_indices, 1)
    mask = mask.view(-1)
    # Shape: (batch_size * sequence_length, 1)
    flattened_target = target.view(-1, 1)
    # Shape: (nonzero_indices, 1)
    unmasked_indices = flattened_indices[mask].unsqueeze(-1)

    flattened_target = flattened_target.scatter(0, unmasked_indices, fill_value)

    filled_target = flattened_target.reshape(prev_shape)

    return filled_target


def masked_index_replace(
    target: torch.Tensor,
    indices: torch.LongTensor,
    mask: torch.BoolTensor,
    replace: torch.Tensor,
) -> torch.Tensor:
    """
    The given `indices` in `target` will be will be replaced with corresponding index
    from the `replace` tensor given a `mask`.


    # Parameters

    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_dim).
        This is the tensor to be replaced into.
    indices : `torch.LongTensor`, required
        A 2 dimensional tensor of shape (batch_size, num_indices),
        These are the indices that will be replaced in the original tensor.
    mask : `torch.Tensor`, required.
        A 2 dimensional tensor of shape (batch_size, num_indices), mask.sum() == `nonzero_indices`.
    replace : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, num_indices, embedding_dim),
        The tensor to perform scatter from.

    # Returns

    replaced_target : `torch.Tensor`
        A tensor with shape (batch_size, sequence_length, embedding_dim) where 'indices'
        are replaced with the corrosponding vector from `replace`
    """
    target = target.clone()
    mask = mask.bool()
    prev_shape = target.size()
    # Shape: (batch_size * num_indices)
    flattened_indices = flatten_and_batch_shift_indices(indices * mask, target.size(1))
    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))
    # Shape: (nonzero_indices, 1)
    mask = mask.view(-1)
    flattened_target[flattened_indices[mask]] = replace.view(-1, replace.size(-1))[mask]
    # Shape: (batch_size, sequence_length, embedding_dim)
    replaced_target = flattened_target.reshape(prev_shape)
    return replaced_target


def batched_span_select(target: torch.Tensor, spans: torch.LongTensor) -> torch.Tensor:
    """
    The given `spans` of size `(batch_size, num_spans, 2)` indexes into the sequence
    dimension (dimension 2) of the target, which has size `(batch_size, sequence_length,
    embedding_size)`.

    This function returns segmented spans in the target with respect to the provided span indices.

    # Parameters

    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : `torch.LongTensor`
        A 3 dimensional tensor of shape (batch_size, num_spans, 2) representing start and end
        indices (both inclusive) into the `sequence_length` dimension of the `target` tensor.

    # Returns

    span_embeddings : `torch.Tensor`
        A tensor with shape (batch_size, num_spans, max_batch_span_width, embedding_size]
        representing the embedded spans extracted from the batch flattened target tensor.
    span_mask: `torch.BoolTensor`
        A tensor with shape (batch_size, num_spans, max_batch_span_width) representing the mask on
        the returned span embeddings.
    """
    # both of shape (batch_size, num_spans, 1)
    span_starts, span_ends = spans.split(1, dim=-1)

    # shape (batch_size, num_spans, 1)
    # These span widths are off by 1, because the span ends are `inclusive`.
    span_widths = span_ends - span_starts

    # We need to know the maximum span width so we can
    # generate indices to extract the spans from the sequence tensor.
    # These indices will then get masked below, such that if the length
    # of a given span is smaller than the max, the rest of the values
    # are masked.
    max_batch_span_width = span_widths.max().item() + 1

    # Shape: (1, 1, max_batch_span_width)
    max_span_range_indices = get_range_vector(max_batch_span_width, get_device_of(target)).view(
        1, 1, -1
    )
    # Shape: (batch_size, num_spans, max_batch_span_width)
    # This is a broadcasted comparison - for each span we are considering,
    # we are creating a range vector of size max_span_width, but masking values
    # which are greater than the actual length of the span.
    #
    # We're using <= here (and for the mask below) because the span ends are
    # inclusive, so we want to include indices which are equal to span_widths rather
    # than using it as a non-inclusive upper bound.
    span_mask = max_span_range_indices <= span_widths
    raw_span_indices = span_starts + max_span_range_indices
    # We also don't want to include span indices which greater than the sequence_length,
    # which happens because some spans near the end of the sequence
    # have a start index + max_batch_span_width > sequence_length, so we add this to the mask here.
    span_mask = span_mask & (raw_span_indices < target.size(1)) & (0 <= raw_span_indices)
    span_indices = raw_span_indices * span_mask

    # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
    span_embeddings = batched_index_select(target, span_indices)

    return span_embeddings, span_mask


def flattened_index_select(target: torch.Tensor, indices: torch.LongTensor) -> torch.Tensor:
    """
    The given `indices` of size `(set_size, subset_size)` specifies subsets of the `target`
    that each of the set_size rows should select. The `target` has size
    `(batch_size, sequence_length, embedding_size)`, and the resulting selected tensor has size
    `(batch_size, set_size, subset_size, embedding_size)`.

    # Parameters

    target : `torch.Tensor`, required.
        A Tensor of shape (batch_size, sequence_length, embedding_size).
    indices : `torch.LongTensor`, required.
        A LongTensor of shape (set_size, subset_size). All indices must be < sequence_length
        as this tensor is an index into the sequence_length dimension of the target.

    # Returns

    selected : `torch.Tensor`, required.
        A Tensor of shape (batch_size, set_size, subset_size, embedding_size).
    """
    if indices.dim() != 2:
        raise ConfigurationError(
            "Indices passed to flattened_index_select had shape {} but "
            "only 2 dimensional inputs are supported.".format(indices.size())
        )
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


def bucket_values(
    distances: torch.Tensor, num_identity_buckets: int = 4, num_total_buckets: int = 10
) -> torch.Tensor:
    """
    Places the given values (designed for distances) into `num_total_buckets`semi-logscale
    buckets, with `num_identity_buckets` of these capturing single values.

    The default settings will bucket values into the following buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].

    # Parameters

    distances : `torch.Tensor`, required.
        A Tensor of any size, to be bucketed.
    num_identity_buckets: `int`, optional (default = `4`).
        The number of identity buckets (those only holding a single value).
    num_total_buckets : `int`, (default = `10`)
        The total number of buckets to bucket values into.

    # Returns

    `torch.Tensor`
        A tensor of the same shape as the input, containing the indices of the buckets
        the values were placed in.
    """
    # Chunk the values into semi-logscale buckets using .floor().
    # This is a semi-logscale bucketing because we divide by log(2) after taking the log.
    # We do this to make the buckets more granular in the initial range, where we expect
    # most values to fall. We then add (num_identity_buckets - 1) because we want these indices
    # to start _after_ the fixed number of buckets which we specified would only hold single values.
    logspace_index = (distances.float().log() / math.log(2)).floor().long() + (
        num_identity_buckets - 1
    )
    # create a mask for values which will go into single number buckets (i.e not a range).
    use_identity_mask = (distances <= num_identity_buckets).long()
    use_buckets_mask = 1 + (-1 * use_identity_mask)
    # Use the original values if they are less than num_identity_buckets, otherwise
    # use the logspace indices.
    combined_index = use_identity_mask * distances + use_buckets_mask * logspace_index
    # Clamp to put anything > num_total_buckets into the final bucket.
    return combined_index.clamp(0, num_total_buckets - 1)


def add_sentence_boundary_token_ids(
    tensor: torch.Tensor, mask: torch.BoolTensor, sentence_begin_token: Any, sentence_end_token: Any
) -> Tuple[torch.Tensor, torch.BoolTensor]:
    """
    Add begin/end of sentence tokens to the batch of sentences.
    Given a batch of sentences with size `(batch_size, timesteps)` or
    `(batch_size, timesteps, dim)` this returns a tensor of shape
    `(batch_size, timesteps + 2)` or `(batch_size, timesteps + 2, dim)` respectively.

    Returns both the new tensor and updated mask.

    # Parameters

    tensor : `torch.Tensor`
        A tensor of shape `(batch_size, timesteps)` or `(batch_size, timesteps, dim)`
    mask : `torch.BoolTensor`
         A tensor of shape `(batch_size, timesteps)`
    sentence_begin_token: `Any`
        Can be anything that can be broadcast in torch for assignment.
        For 2D input, a scalar with the `<S>` id. For 3D input, a tensor with length dim.
    sentence_end_token: `Any`
        Can be anything that can be broadcast in torch for assignment.
        For 2D input, a scalar with the `</S>` id. For 3D input, a tensor with length dim.

    # Returns

    tensor_with_boundary_tokens : `torch.Tensor`
        The tensor with the appended and prepended boundary tokens. If the input was 2D,
        it has shape (batch_size, timesteps + 2) and if the input was 3D, it has shape
        (batch_size, timesteps + 2, dim).
    new_mask : `torch.BoolTensor`
        The new mask for the tensor, taking into account the appended tokens
        marking the beginning and end of the sentence.
    """
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    tensor_shape = list(tensor.data.shape)
    new_shape = list(tensor_shape)
    new_shape[1] = tensor_shape[1] + 2
    tensor_with_boundary_tokens = tensor.new_zeros(*new_shape, device=tensor.device)
    if len(tensor_shape) == 2:
        tensor_with_boundary_tokens[:, 1:-1] = tensor
        tensor_with_boundary_tokens[:, 0] = sentence_begin_token
        for i, j in enumerate(sequence_lengths):
            tensor_with_boundary_tokens[i, j + 1] = sentence_end_token
        new_mask = tensor_with_boundary_tokens != 0
    elif len(tensor_shape) == 3:
        tensor_with_boundary_tokens[:, 1:-1, :] = tensor
        sentence_begin_token = sentence_begin_token.detach().to(tensor.device)
        sentence_end_token = sentence_end_token.detach().to(tensor.device)
        for i, j in enumerate(sequence_lengths):
            tensor_with_boundary_tokens[i, 0, :] = sentence_begin_token
            tensor_with_boundary_tokens[i, j + 1, :] = sentence_end_token
        new_mask = (tensor_with_boundary_tokens > 0).sum(dim=-1) > 0
    else:
        raise ValueError("add_sentence_boundary_token_ids only accepts 2D and 3D input")

    return tensor_with_boundary_tokens, new_mask


def remove_sentence_boundaries(
    tensor: torch.Tensor, mask: torch.BoolTensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove begin/end of sentence embeddings from the batch of sentences.
    Given a batch of sentences with size `(batch_size, timesteps, dim)`
    this returns a tensor of shape `(batch_size, timesteps - 2, dim)` after removing
    the beginning and end sentence markers.  The sentences are assumed to be padded on the right,
    with the beginning of each sentence assumed to occur at index 0 (i.e., `mask[:, 0]` is assumed
    to be 1).

    Returns both the new tensor and updated mask.

    This function is the inverse of `add_sentence_boundary_token_ids`.

    # Parameters

    tensor : `torch.Tensor`
        A tensor of shape `(batch_size, timesteps, dim)`
    mask : `torch.BoolTensor`
         A tensor of shape `(batch_size, timesteps)`

    # Returns

    tensor_without_boundary_tokens : `torch.Tensor`
        The tensor after removing the boundary tokens of shape `(batch_size, timesteps - 2, dim)`
    new_mask : `torch.BoolTensor`
        The new mask for the tensor of shape `(batch_size, timesteps - 2)`.
    """
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    tensor_shape = list(tensor.data.shape)
    new_shape = list(tensor_shape)
    new_shape[1] = tensor_shape[1] - 2
    tensor_without_boundary_tokens = tensor.new_zeros(*new_shape)
    new_mask = tensor.new_zeros((new_shape[0], new_shape[1]), dtype=torch.bool)
    for i, j in enumerate(sequence_lengths):
        if j > 2:
            tensor_without_boundary_tokens[i, : (j - 2), :] = tensor[i, 1 : (j - 1), :]
            new_mask[i, : (j - 2)] = True

    return tensor_without_boundary_tokens, new_mask


def add_positional_features(
    tensor: torch.Tensor, min_timescale: float = 1.0, max_timescale: float = 1.0e4
):

    """
    Implements the frequency-based positional encoding described
    in [Attention is All you Need][0].

    Adds sinusoids of different frequencies to a `Tensor`. A sinusoid of a
    different frequency and phase is added to each dimension of the input `Tensor`.
    This allows the attention heads to use absolute and relative positions.

    The number of timescales is equal to hidden_dim / 2 within the range
    (min_timescale, max_timescale). For each timescale, the two sinusoidal
    signals sin(timestep / timescale) and cos(timestep / timescale) are
    generated and concatenated along the hidden_dim dimension.

    [0]: https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077

    # Parameters

    tensor : `torch.Tensor`
        a Tensor with shape (batch_size, timesteps, hidden_dim).
    min_timescale : `float`, optional (default = `1.0`)
        The smallest timescale to use.
    max_timescale : `float`, optional (default = `1.0e4`)
        The largest timescale to use.

    # Returns

    `torch.Tensor`
        The input tensor augmented with the sinusoidal frequencies.
    """  # noqa
    _, timesteps, hidden_dim = tensor.size()

    timestep_range = get_range_vector(timesteps, get_device_of(tensor)).data.float()
    # We're generating both cos and sin frequencies,
    # so half for each.
    num_timescales = hidden_dim // 2
    timescale_range = get_range_vector(num_timescales, get_device_of(tensor)).data.float()

    log_timescale_increments = math.log(float(max_timescale) / float(min_timescale)) / float(
        num_timescales - 1
    )
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
    return torch.nn.ModuleList(copy.deepcopy(module) for _ in range(num_copies))


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


def inspect_parameters(module: torch.nn.Module, quiet: bool = False) -> Dict[str, Any]:
    """
    Inspects the model/module parameters and their tunability. The output is structured
    in a nested dict so that parameters in same sub-modules are grouped together.
    This can be helpful to setup module path based regex, for example in initializer.
    It prints it by default (optional) and returns the inspection dict. Eg. output::

        {
            "_text_field_embedder": {
                "token_embedder_tokens": {
                    "_projection": {
                        "bias": "tunable",
                        "weight": "tunable"
                    },
                    "weight": "frozen"
                }
            }
        }

    """
    results: Dict[str, Any] = {}
    for name, param in sorted(module.named_parameters()):
        keys = name.split(".")
        write_to = results
        for key in keys[:-1]:
            if key not in write_to:
                write_to[key] = {}
            write_to = write_to[key]
        write_to[keys[-1]] = "tunable" if param.requires_grad else "frozen"
    if not quiet:
        print(json.dumps(results, indent=4))
    return results


def find_text_field_embedder(model: torch.nn.Module) -> torch.nn.Module:
    """
    Takes a `Model` and returns the `Module` that is a `TextFieldEmbedder`.  We return just the
    first one, as it's very rare to have more than one.  If there isn't a `TextFieldEmbedder` in the
    given `Model`, we raise a `ValueError`.
    """
    from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder

    for module in model.modules():
        if isinstance(module, TextFieldEmbedder):
            return module
    raise ValueError("Couldn't find TextFieldEmbedder!")


def find_embedding_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    Takes a model (typically an AllenNLP `Model`, but this works for any `torch.nn.Module`) and
    makes a best guess about which module is the embedding layer.  For typical AllenNLP models,
    this often is the `TextFieldEmbedder`, but if you're using a pre-trained contextualizer, we
    really want layer 0 of that contextualizer, not the output.  So there are a bunch of hacks in
    here for specific pre-trained contextualizers.
    """
    # We'll look for a few special cases in a first pass, then fall back to just finding a
    # TextFieldEmbedder in a second pass if we didn't find a special case.
    from transformers.models.gpt2.modeling_gpt2 import GPT2Model
    from transformers.models.bert.modeling_bert import BertEmbeddings
    from transformers.models.albert.modeling_albert import AlbertEmbeddings
    from transformers.models.roberta.modeling_roberta import RobertaEmbeddings
    from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
    from allennlp.modules.text_field_embedders.basic_text_field_embedder import (
        BasicTextFieldEmbedder,
    )
    from allennlp.modules.token_embedders.embedding import Embedding

    for module in model.modules():
        if isinstance(module, BertEmbeddings):
            return module.word_embeddings
        if isinstance(module, RobertaEmbeddings):
            return module.word_embeddings
        if isinstance(module, AlbertEmbeddings):
            return module.word_embeddings
        if isinstance(module, GPT2Model):
            return module.wte

    for module in model.modules():
        if isinstance(module, TextFieldEmbedder):

            if isinstance(module, BasicTextFieldEmbedder):
                # We'll have a check for single Embedding cases, because we can be more efficient
                # in cases like this.  If this check fails, then for something like hotflip we need
                # to actually run the text field embedder and construct a vector for each token.
                if len(module._token_embedders) == 1:
                    embedder = list(module._token_embedders.values())[0]
                    if isinstance(embedder, Embedding):
                        if embedder._projection is None:
                            # If there's a projection inside the Embedding, then we need to return
                            # the whole TextFieldEmbedder, because there's more computation that
                            # needs to be done than just multiply by an embedding matrix.
                            return embedder
            return module
    raise RuntimeError("No embedding module found!")


def get_token_offsets_from_text_field_inputs(
    text_field_inputs: List[Any],
) -> Optional[torch.Tensor]:
    """
    Given a list of inputs to a TextFieldEmbedder, tries to find token offsets from those inputs, if
    there are any.  You will have token offsets if you are using a mismatched token embedder; if
    you're not, the return value from this function should be None.  This function is intended to be
    called from a `forward_hook` attached to a `TextFieldEmbedder`, so the inputs are formatted just
    as a list.

    It's possible in theory that you could have multiple offsets as inputs to a single call to a
    `TextFieldEmbedder`, but that's an extremely rare use case (I can't really imagine anyone
    wanting to do that).  In that case, we'll only return the first one.  If you need different
    behavior for your model, open an issue on github describing what you're doing.
    """
    for input_index, text_field_input in enumerate(text_field_inputs):
        if not isinstance(text_field_input, dict):
            continue
        for input_value in text_field_input.values():
            if not isinstance(input_value, dict):
                continue
            for embedder_arg_name, embedder_arg_value in input_value.items():
                if embedder_arg_name == "offsets":
                    return embedder_arg_value
    return None


def extend_layer(layer: torch.nn.Module, new_dim: int) -> None:
    valid_layers = [torch.nn.Linear, torch.nn.Bilinear]
    if not any([isinstance(layer, i) for i in valid_layers]):
        raise ConfigurationError("Inappropriate layer type")

    extend_dim = new_dim - layer.out_features
    if not extend_dim:
        return layer

    if isinstance(layer, torch.nn.Linear):
        new_weight = torch.FloatTensor(extend_dim, layer.in_features)
    elif isinstance(layer, torch.nn.Bilinear):
        new_weight = torch.FloatTensor(extend_dim, layer.in1_features, layer.in2_features)

    new_bias = torch.FloatTensor(extend_dim)
    torch.nn.init.xavier_uniform_(new_weight)
    torch.nn.init.zeros_(new_bias)

    device = layer.weight.device
    layer.weight = torch.nn.Parameter(
        torch.cat([layer.weight.data, new_weight.to(device)], dim=0),
        requires_grad=layer.weight.requires_grad,
    )
    layer.bias = torch.nn.Parameter(
        torch.cat([layer.bias.data, new_bias.to(device)], dim=0),
        requires_grad=layer.bias.requires_grad,
    )
    layer.out_features = new_dim


def masked_topk(
    input_: torch.FloatTensor,
    mask: torch.BoolTensor,
    k: Union[int, torch.LongTensor],
    dim: int = -1,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor]:
    """
    Extracts the top-k items along a certain dimension. This is similar to `torch.topk` except:
    (1) we allow of a `mask` that makes the function not consider certain elements;
    (2) the returned top input, mask, and indices are sorted in their original order in the input;
    (3) May use the same k for all dimensions, or different k for each.

    # Parameters

    input_ : `torch.FloatTensor`, required.
        A tensor containing the items that we want to prune.
    mask : `torch.BoolTensor`, required.
        A tensor with the same shape as `input_` that makes the function not consider masked out
        (i.e. False) elements.
    k : `Union[int, torch.LongTensor]`, required.
        If a tensor of shape as `input_` except without dimension `dim`, specifies the number of
        items to keep for each dimension.
        If an int, keep the same number of items for all dimensions.

    # Returns

    top_input : `torch.FloatTensor`
        The values of the top-k scoring items.
        Has the same shape as `input_` except dimension `dim` has value `k` when it's an `int`
        or `k.max()` when it's a tensor.
    top_mask : `torch.BoolTensor`
        The corresponding mask for `top_input`.
        Has the shape as `top_input`.
    top_indices : `torch.IntTensor`
        The indices of the top-k scoring items into the original `input_`
        tensor. This is returned because it can be useful to retain pointers to
        the original items, if each item is being scored by multiple distinct
        scorers, for instance.
        Has the shape as `top_input`.
    """
    if input_.size() != mask.size():
        raise ValueError("`input_` and `mask` must have the same shape.")
    if not -input_.dim() <= dim < input_.dim():
        raise ValueError("`dim` must be in `[-input_.dim(), input_.dim())`")
    dim = (dim + input_.dim()) % input_.dim()

    max_k = k if isinstance(k, int) else k.max()

    # We put the dim in question to the last dimension by permutation, and squash all leading dims.

    # [0, 1, ..., dim - 1, dim + 1, ..., input.dim() - 1, dim]
    permutation = list(range(input_.dim()))
    permutation.pop(dim)
    permutation += [dim]

    # [0, 1, ..., dim - 1, -1, dim, ..., input.dim() - 2]; for restoration
    reverse_permutation = list(range(input_.dim() - 1))
    reverse_permutation.insert(dim, -1)

    other_dims_size = list(input_.size())
    other_dims_size.pop(dim)
    permuted_size = other_dims_size + [max_k]  # for restoration

    # If an int was given for number of items to keep, construct tensor by repeating the value.
    if isinstance(k, int):
        # Put the tensor on same device as the mask.
        k = k * torch.ones(*other_dims_size, dtype=torch.long, device=mask.device)
    else:
        if list(k.size()) != other_dims_size:
            raise ValueError(
                "`k` must have the same shape as `input_` with dimension `dim` removed."
            )

    num_items = input_.size(dim)
    # (batch_size, num_items)  -- "batch_size" refers to all other dimensions stacked together
    input_ = input_.permute(*permutation).reshape(-1, num_items)
    mask = mask.permute(*permutation).reshape(-1, num_items)
    k = k.reshape(-1)

    # Make sure that we don't select any masked items by setting their scores to be very
    # negative.
    input_ = replace_masked_values(input_, mask, min_value_of_dtype(input_.dtype))

    # Shape: (batch_size, max_k)
    _, top_indices = input_.topk(max_k, 1)

    # Mask based on number of items to keep for each sentence.
    # Shape: (batch_size, max_k)
    top_indices_mask = get_mask_from_sequence_lengths(k, max_k).bool()

    # Fill all masked indices with largest "top" index for that sentence, so that all masked
    # indices will be sorted to the end.
    # Shape: (batch_size, 1)
    fill_value, _ = top_indices.max(dim=1, keepdim=True)
    # Shape: (batch_size, max_num_items_to_keep)
    top_indices = torch.where(top_indices_mask, top_indices, fill_value)

    # Now we order the selected indices in increasing order with
    # respect to their indices (and hence, with respect to the
    # order they originally appeared in the `embeddings` tensor).
    top_indices, _ = top_indices.sort(1)

    # Combine the masks on spans that are out-of-bounds, and the mask on spans that are outside
    # the top k for each sentence.
    # Shape: (batch_size, max_k)
    sequence_mask = mask.gather(1, top_indices)
    top_mask = top_indices_mask & sequence_mask

    # Shape: (batch_size, max_k)
    top_input = input_.gather(1, top_indices)

    return (
        top_input.reshape(*permuted_size).permute(*reverse_permutation),
        top_mask.reshape(*permuted_size).permute(*reverse_permutation),
        top_indices.reshape(*permuted_size).permute(*reverse_permutation),
    )


def info_value_of_dtype(dtype: torch.dtype):
    """
    Returns the `finfo` or `iinfo` object of a given PyTorch data type. Does not allow torch.bool.
    """
    if dtype == torch.bool:
        raise TypeError("Does not support torch.bool")
    elif dtype.is_floating_point:
        return torch.finfo(dtype)
    else:
        return torch.iinfo(dtype)


def min_value_of_dtype(dtype: torch.dtype):
    """
    Returns the minimum value of a given PyTorch data type. Does not allow torch.bool.
    """
    return info_value_of_dtype(dtype).min


def max_value_of_dtype(dtype: torch.dtype):
    """
    Returns the maximum value of a given PyTorch data type. Does not allow torch.bool.
    """
    return info_value_of_dtype(dtype).max


def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


_V = TypeVar("_V", int, float, torch.Tensor)


def distributed_device() -> torch.device:
    """
    Get the correct `torch.device` of the current process to use for distributed point-to-point communication.
    """
    if not is_distributed():
        raise RuntimeError(
            "'distributed_device()' can only be called within a distributed process group"
        )
    return int_to_device(-1 if dist.get_backend() != "nccl" else torch.cuda.current_device())


def dist_reduce(value: _V, reduce_op, **kwargs) -> _V:
    """
    Reduces the given `value` across all distributed worker nodes according the given
    reduction operation.

    If called outside of a distributed context, it will just return `value`.

    # Parameters

    value : `_V`
        The value to reduce across distributed nodes.
    reduce_op : `torch.distributed.ReduceOp`
        The [reduction operation](https://pytorch.org/docs/stable/distributed.html#torch.distributed.ReduceOp)
        to use.
    **kwargs : `Any`
        Additional arguments used to construct the tensor that will wrap `value`.

    # Returns

    `_V`
        The final value.
    """
    if not is_distributed():
        return value
    device = distributed_device()
    value_tensor = torch.tensor(value, device=device, **kwargs)
    dist.all_reduce(value_tensor, op=reduce_op)

    if isinstance(value, torch.Tensor):
        return value_tensor
    return value_tensor.item()  # type: ignore[return-value]


def dist_reduce_sum(value: _V, **kwargs) -> _V:
    """
    Sums the given `value` across distributed worker nodes.
    This is equivalent to calling `dist_reduce(v, dist.ReduceOp.SUM)`.
    """
    # NOTE: Why have this check here even though the same check is in `dist_reduce()`?
    # Because we want to be able to call this function even when torch's distributed framework
    # is not available...
    # If torch's distributed framework is not available on the system, then `torch.distributed`
    # (imported here as `dist`) will just be an empty module. So calling `dist.ReduceOp.SUM` would
    # result in an `AttributeError`.
    if not is_distributed():
        return value
    return dist_reduce(value, dist.ReduceOp.SUM, **kwargs)


def _collect_state_dict(
    module: torch.nn.Module, state_dict: Optional[StateDictType], recurse: bool = True
) -> Tuple[StateDictType, List[str], List[str]]:
    """
    Collect a module's state dict across distributed processes.

    Returns the syncronized state dictionary, which will always be a valid state dict,
    and then the missing and unexpected keys corresponding to the original `state_dict`.
    Parameters that missing from the original `state_dict` will be populated from the
    corresponding parameter in the primary processes' module's state dict.

    !!! Note

        `missing_keys` and `unexpected_keys` are only populated in the primary process.
    """
    # This is the device we'll use for the broadcast operation.
    dist_device = distributed_device()
    # This is the device we'll put all tensors on in the returned state dict.
    state_dict_device = (
        int_to_device(-1) if not state_dict else state_dict[list(state_dict.keys())[0]].device
    )

    missing_keys: List[str] = []
    unexpected_keys: List[str] = []

    # Gather current state dict and prepare to iterator over it.
    # We iterate over this state dict instead of `state_dict` so we can be sure
    # that the order is consistent across processes.
    # We'll also update this state dict as we go and return it at the end.
    if recurse:
        current_state_dict = module.state_dict()
    else:
        # Only collect state of direct members, including both parameters and buffers.
        current_state_dict = OrderedDict(
            chain(
                # Paramaters
                ((n, p.data) for (n, p) in module.named_parameters(recurse=False)),
                # Buffers
                module.named_buffers(recurse=False),
            )
        )

    keys = list(current_state_dict.keys())

    # Gather unexpected_keys.
    if is_global_primary():
        assert state_dict is not None
        module_keys = set(module.state_dict().keys())
        for key in state_dict:
            if key not in module_keys:
                unexpected_keys.append(key)

    for key in keys:
        tensor = current_state_dict[key]
        if is_global_primary():
            assert state_dict is not None
            if key in state_dict:
                # Update `tensor` to the value in `state_dict`.
                tensor = state_dict[key]
            else:
                missing_keys.append(key)
        tensor = tensor.to(dist_device)
        dist.broadcast(tensor, 0)
        current_state_dict[key] = tensor.to(state_dict_device)

    return current_state_dict, missing_keys, unexpected_keys


class _LoadStateDictResult(NamedTuple):
    missing_keys: List[str]
    unexpected_keys: List[str]


def load_state_dict_distributed(
    module: torch.nn.Module, state_dict: Optional[StateDictType], strict: bool = True
) -> _LoadStateDictResult:
    """
    Load a `state_dict` to the `module` within a distributed process. Only the global
    primary process requires the `state_dict` to not be `None`. All other processes
    will have the state tensors broadcasted to them one-by-one.

    If `strict` is `True`, then the keys of `state_dict` must exactly match the keys
    returned by `module.state_dict()`.

    !!! Note
        The returned `missing_keys` and `unexpected_keys` will only be accurate
        in the primary process.

    # Returns

    `_LoadStateDictResult`
        A `NamedTuple` with `missing_keys` and `unexpected_keys` fields, both of which
        are lists of strings.

    # Raises

    `RuntimeError`
        If `strict` is `True` and there are missing or unexpected keys.

    """
    if not is_distributed():
        return module.load_state_dict(state_dict, strict=strict)

    if is_global_primary():
        assert state_dict is not None
    else:
        assert state_dict is None

    missing_keys: List[str] = []
    unexpected_keys: List[str] = []

    submodules = dict(module.named_children())

    def update_key_list(original, updates):
        for key in updates:
            if key not in original:
                original.append(key)

    # If we've found a sharded module or there aren't any more submodules of the current module,
    # we collect the state_dict and load it now instead of recursing further.
    if getattr(module, _MODULE_SHARDED_FLAG, False) or not submodules:
        # Collect.
        state_dict, _missing_keys, _unexpected_keys = _collect_state_dict(module, state_dict)
        assert state_dict is not None
        update_key_list(missing_keys, _missing_keys)
        update_key_list(unexpected_keys, _unexpected_keys)
        # And load.
        _missing_keys, _unexpected_keys = module.load_state_dict(state_dict, strict=False)
        update_key_list(missing_keys, _missing_keys)
        update_key_list(unexpected_keys, _unexpected_keys)
    else:
        # We'll recursively call this function on each submodule, but first we need
        # to collect any parameters that are direct members of this module.
        direct_member_state_dict, _missing_keys, _unexpected_keys = _collect_state_dict(
            module, state_dict, recurse=False
        )
        update_key_list(missing_keys, _missing_keys)
        update_key_list(unexpected_keys, _unexpected_keys)

        # `_missing_keys` here will contain any keys corresponding to submodules, but
        # we'll remove those below.
        _missing_keys, _unexpected_keys = module.load_state_dict(
            direct_member_state_dict, strict=False
        )
        update_key_list(missing_keys, _missing_keys)
        update_key_list(unexpected_keys, _unexpected_keys)

        # Okay, now for the recursive part.
        for name, submodule in submodules.items():
            # Update `missing_keys` to remove keys corresponding to this submodule.
            # If they are actually missing after this step, we add them back in below.
            missing_keys = [k for k in missing_keys if not k.startswith(name + ".")]
            submodule_state_dict: Optional[StateDictType] = None
            if is_global_primary():
                assert state_dict is not None
                submodule_state_dict = {
                    key.replace(name + ".", "", 1): value
                    for key, value in state_dict.items()
                    if key.startswith(name + ".")
                }
            _missing_keys, _unexpected_keys = load_state_dict_distributed(
                submodule, submodule_state_dict, strict=False
            )
            update_key_list(missing_keys, [f"{name}.{key}" for key in _missing_keys])
            update_key_list(unexpected_keys, [f"{name}.{key}" for key in _unexpected_keys])

    if strict:
        error_msgs: List[str] = []
        if missing_keys:
            error_msgs.append(
                "Missing key(s) in state_dict: {}".format(", ".join(f'"{k}"' for k in missing_keys))
            )
        if unexpected_keys:
            error_msgs.append(
                "Unexpected key(s) in state_dict: {}".format(
                    ", ".join(f'"{k}"' for k in unexpected_keys)
                )
            )
        if error_msgs:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    module.__class__.__name__, "\n\t".join(error_msgs)
                )
            )

    return _LoadStateDictResult(missing_keys, unexpected_keys)
