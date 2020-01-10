# allennlp.nn.util

Assorted utilities for working with neural networks in AllenNLP.

## has_tensor
```python
has_tensor(obj) -> bool
```

Given a possibly complex data structure,
check if it has any torch.Tensors in it.

## move_to_device
```python
move_to_device(obj, cuda_device:int)
```

Given a structure (possibly) containing Tensors on the CPU,
move all the Tensors to the specified GPU (or do nothing, if they should be on the CPU).

## clamp_tensor
```python
clamp_tensor(tensor, minimum, maximum)
```

Supports sparse and dense tensors.
Returns a tensor with values clamped between the provided minimum and maximum,
without modifying the original tensor.

## batch_tensor_dicts
```python
batch_tensor_dicts(tensor_dicts:List[Dict[str, torch.Tensor]], remove_trailing_dimension:bool=False) -> Dict[str, torch.Tensor]
```

Takes a list of tensor dictionaries, where each dictionary is assumed to have matching keys,
and returns a single dictionary with all tensors with the same key batched together.

Parameters
----------
tensor_dicts : ``List[Dict[str, torch.Tensor]]``
    The list of tensor dictionaries to batch.
remove_trailing_dimension : ``bool``
    If ``True``, we will check for a trailing dimension of size 1 on the tensors that are being
    batched, and remove it if we find it.

## get_lengths_from_binary_sequence_mask
```python
get_lengths_from_binary_sequence_mask(mask:torch.Tensor)
```

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

## get_mask_from_sequence_lengths
```python
get_mask_from_sequence_lengths(sequence_lengths:torch.Tensor, max_length:int) -> torch.Tensor
```

Given a variable of shape ``(batch_size,)`` that represents the sequence lengths of each batch
element, this function returns a ``(batch_size, max_length)`` mask variable.  For example, if
our input was ``[2, 2, 3]``, with a ``max_length`` of 4, we'd return
``[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]``.

We require ``max_length`` here instead of just computing it from the input ``sequence_lengths``
because it lets us avoid finding the max, then copying that value from the GPU to the CPU so
that we can use it to construct a new tensor.

## sort_batch_by_length
```python
sort_batch_by_length(tensor:torch.Tensor, sequence_lengths:torch.Tensor)
```

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
permutation_index : torch.LongTensor
    The indices used to sort the tensor. This is useful if you want to sort many
    tensors using the same ordering.

## get_final_encoder_states
```python
get_final_encoder_states(encoder_outputs:torch.Tensor, mask:torch.Tensor, bidirectional:bool=False) -> torch.Tensor
```

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

## get_dropout_mask
```python
get_dropout_mask(dropout_probability:float, tensor_for_masking:torch.Tensor)
```

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

## masked_softmax
```python
masked_softmax(vector:torch.Tensor, mask:torch.Tensor, dim:int=-1, memory_efficient:bool=False, mask_fill_value:float=-1e+32) -> torch.Tensor
```

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

## masked_log_softmax
```python
masked_log_softmax(vector:torch.Tensor, mask:torch.Tensor, dim:int=-1) -> torch.Tensor
```

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

## masked_max
```python
masked_max(vector:torch.Tensor, mask:torch.Tensor, dim:int, keepdim:bool=False, min_val:float=-10000000.0) -> torch.Tensor
```

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

## masked_mean
```python
masked_mean(vector:torch.Tensor, mask:torch.Tensor, dim:int, keepdim:bool=False, eps:float=1e-08) -> torch.Tensor
```

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

## masked_flip
```python
masked_flip(padded_sequence:torch.Tensor, sequence_lengths:List[int]) -> torch.Tensor
```

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

## viterbi_decode
```python
viterbi_decode(tag_sequence:torch.Tensor, transition_matrix:torch.Tensor, tag_observations:Union[List[int], NoneType]=None, allowed_start_transitions:torch.Tensor=None, allowed_end_transitions:torch.Tensor=None, top_k:int=None)
```

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
allowed_start_transitions : torch.Tensor, optional, (default = None)
    An optional tensor of shape (num_tags,) describing which tags the START token
    may transition *to*. If provided, additional transition constraints will be used for
    determining the start element of the sequence.
allowed_end_transitions : torch.Tensor, optional, (default = None)
    An optional tensor of shape (num_tags,) describing which tags may transition *to* the
    end tag. If provided, additional transition constraints will be used for determining
    the end element of the sequence.
top_k : int, optional, (default = None)
    Optional integer specifying how many of the top paths to return. For top_k>=1, returns
    a tuple of two lists: top_k_paths, top_k_scores, For top_k==None, returns a flattened
    tuple with just the top path and its score (not in lists, for backwards compatibility).

Returns
-------
viterbi_path : List[int]
    The tag indices of the maximum likelihood tag sequence.
viterbi_score : torch.Tensor
    The score of the viterbi path.

## get_text_field_mask
```python
get_text_field_mask(text_field_tensors:Dict[str, torch.Tensor], num_wrapping_dims:int=0) -> torch.LongTensor
```

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

## weighted_sum
```python
weighted_sum(matrix:torch.Tensor, attention:torch.Tensor) -> torch.Tensor
```

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

## sequence_cross_entropy_with_logits
```python
sequence_cross_entropy_with_logits(logits:torch.FloatTensor, targets:torch.LongTensor, weights:torch.FloatTensor, average:str='batch', label_smoothing:float=None, gamma:float=None, alpha:Union[float, List[float], torch.FloatTensor]=None) -> torch.FloatTensor
```

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
gamma : ``float``, optional (default = None)
    Focal loss[*] focusing parameter ``gamma`` to reduces the relative loss for
    well-classified examples and put more focus on hard. The greater value
    ``gamma`` is, the more focus on hard examples.
alpha : ``float`` or ``List[float]``, optional (default = None)
    Focal loss[*] weighting factor ``alpha`` to balance between classes. Can be
    used independently with ``gamma``. If a single ``float`` is provided, it
    is assumed binary case using ``alpha`` and ``1 - alpha`` for positive and
    negative respectively. If a list of ``float`` is provided, with the same
    length as the number of classes, the weights will match the classes.
    [*] T. Lin, P. Goyal, R. Girshick, K. He and P. Dollár, "Focal Loss for
    Dense Object Detection," 2017 IEEE International Conference on Computer
    Vision (ICCV), Venice, 2017, pp. 2999-3007.

Returns
-------
A torch.FloatTensor representing the cross entropy loss.
If ``average=="batch"`` or ``average=="token"``, the returned loss is a scalar.
If ``average is None``, the returned loss is a vector of shape (batch_size,).


## replace_masked_values
```python
replace_masked_values(tensor:torch.Tensor, mask:torch.Tensor, replace_with:float) -> torch.Tensor
```

Replaces all masked values in ``tensor`` with ``replace_with``.  ``mask`` must be broadcastable
to the same shape as ``tensor``. We require that ``tensor.dim() == mask.dim()``, as otherwise we
won't know which dimensions of the mask to unsqueeze.

This just does ``tensor.masked_fill()``, except the pytorch method fills in things with a mask
value of 1, where we want the opposite.  You can do this in your own code with
``tensor.masked_fill((1 - mask).to(dtype=torch.bool), replace_with)``.

## tensors_equal
```python
tensors_equal(tensor1:torch.Tensor, tensor2:torch.Tensor, tolerance:float=1e-12) -> bool
```

A check for tensor equality (by value).  We make sure that the tensors have the same shape,
then check all of the entries in the tensor for equality.  We additionally allow the input
tensors to be lists or dictionaries, where we then do the above check on every position in the
list / item in the dictionary.  If we find objects that aren't tensors as we're doing that, we
just defer to their equality check.

This is kind of a catch-all method that's designed to make implementing ``__eq__`` methods
easier, in a way that's really only intended to be useful for tests.

## device_mapping
```python
device_mapping(cuda_device:int)
```

In order to `torch.load()` a GPU-trained model onto a CPU (or specific GPU),
you have to supply a `map_location` function. Call this with
the desired `cuda_device` to get the function that `torch.load()` needs.

## combine_tensors
```python
combine_tensors(combination:str, tensors:List[torch.Tensor]) -> torch.Tensor
```

Combines a list of tensors using element-wise operations and concatenation, specified by a
``combination`` string.  The string refers to (1-indexed) positions in the input tensor list,
and looks like ``"1,2,1+2,3-1"``.

We allow the following kinds of combinations : ``x``, ``x*y``, ``x+y``, ``x-y``, and ``x/y``,
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

## combine_tensors_and_multiply
```python
combine_tensors_and_multiply(combination:str, tensors:List[torch.Tensor], weights:torch.nn.parameter.Parameter) -> torch.Tensor
```

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

## get_combined_dim
```python
get_combined_dim(combination:str, tensor_dims:List[int]) -> int
```

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

## logsumexp
```python
logsumexp(tensor:torch.Tensor, dim:int=-1, keepdim:bool=False) -> torch.Tensor
```

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

## get_device_of
```python
get_device_of(tensor:torch.Tensor) -> int
```

Returns the device of the tensor.

## flatten_and_batch_shift_indices
```python
flatten_and_batch_shift_indices(indices:torch.Tensor, sequence_length:int) -> torch.Tensor
```

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

## batched_index_select
```python
batched_index_select(target:torch.Tensor, indices:torch.LongTensor, flattened_indices:Union[torch.LongTensor, NoneType]=None) -> torch.Tensor
```

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

## flattened_index_select
```python
flattened_index_select(target:torch.Tensor, indices:torch.LongTensor) -> torch.Tensor
```

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

## get_range_vector
```python
get_range_vector(size:int, device:int) -> torch.Tensor
```

Returns a range vector with the desired size, starting at 0. The CUDA implementation
is meant to avoid copy data from CPU to GPU.

## bucket_values
```python
bucket_values(distances:torch.Tensor, num_identity_buckets:int=4, num_total_buckets:int=10) -> torch.Tensor
```

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

## add_sentence_boundary_token_ids
```python
add_sentence_boundary_token_ids(tensor:torch.Tensor, mask:torch.Tensor, sentence_begin_token:Any, sentence_end_token:Any) -> Tuple[torch.Tensor, torch.Tensor]
```

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

## remove_sentence_boundaries
```python
remove_sentence_boundaries(tensor:torch.Tensor, mask:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

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

## add_positional_features
```python
add_positional_features(tensor:torch.Tensor, min_timescale:float=1.0, max_timescale:float=10000.0)
```

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

## clone
```python
clone(module:torch.nn.modules.module.Module, num_copies:int) -> torch.nn.modules.container.ModuleList
```
Produce N identical layers.
## combine_initial_dims
```python
combine_initial_dims(tensor:torch.Tensor) -> torch.Tensor
```

Given a (possibly higher order) tensor of ids with shape
(d1, ..., dn, sequence_length)
Return a view that's (d1 * ... * dn, sequence_length).
If original tensor is 1-d or 2-d, return it as is.

## uncombine_initial_dims
```python
uncombine_initial_dims(tensor:torch.Tensor, original_size:torch.Size) -> torch.Tensor
```

Given a tensor of embeddings with shape
(d1 * ... * dn, sequence_length, embedding_dim)
and the original shape
(d1, ..., dn, sequence_length),
return the reshaped tensor of embeddings with shape
(d1, ..., dn, sequence_length, embedding_dim).
If original size is 1-d or 2-d, return it as is.

## inspect_parameters
```python
inspect_parameters(module:torch.nn.modules.module.Module, quiet:bool=False) -> Dict[str, Any]
```

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


## find_embedding_layer
```python
find_embedding_layer(model:torch.nn.modules.module.Module) -> torch.nn.modules.module.Module
```

Takes a model (typically an AllenNLP ``Model``, but this works for any ``torch.nn.Module``) and
makes a best guess about which module is the embedding layer.  For typical AllenNLP models,
this often is the ``TextFieldEmbedder``, but if you're using a pre-trained contextualizer, we
really want layer 0 of that contextualizer, not the output.  So there are a bunch of hacks in
here for specific pre-trained contextualizers.

