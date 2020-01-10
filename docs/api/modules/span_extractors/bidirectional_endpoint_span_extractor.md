# allennlp.modules.span_extractors.bidirectional_endpoint_span_extractor

## BidirectionalEndpointSpanExtractor
```python
BidirectionalEndpointSpanExtractor(self, input_dim:int, forward_combination:str='y-x', backward_combination:str='x-y', num_width_embeddings:int=None, span_width_embedding_dim:int=None, bucket_widths:bool=False, use_sentinels:bool=True) -> None
```

Represents spans from a bidirectional encoder as a concatenation of two different
representations of the span endpoints, one for the forward direction of the encoder
and one from the backward direction. This type of representation encodes some subtlety,
because when you consider the forward and backward directions separately, the end index
of the span for the backward direction's representation is actually the start index.

By default, this ``SpanExtractor`` represents spans as
``sequence_tensor[inclusive_span_end] - sequence_tensor[exclusive_span_start]``
meaning that the representation is the difference between the the last word in the span
and the word `before` the span started. Note that the start and end indices are with
respect to the direction that the RNN is going in, so for the backward direction, the
start/end indices are reversed.

Additionally, the width of the spans can be embedded and concatenated on to the
final combination.

The following other types of representation are supported for both the forward and backward
directions, assuming that ``x = span_start_embeddings`` and ``y = span_end_embeddings``.

``x``, ``y``, ``x*y``, ``x+y``, ``x-y``, ``x/y``, where each of those binary operations
is performed elementwise.  You can list as many combinations as you want, comma separated.
For example, you might give ``x,y,x*y`` as the ``combination`` parameter to this class.
The computed similarity function would then be ``[x; y; x*y]``, which can then be optionally
concatenated with an embedded representation of the width of the span.

Parameters
----------
input_dim : ``int``, required
    The final dimension of the ``sequence_tensor``.
forward_combination : ``str``, optional (default = "y-x").
    The method used to combine the ``forward_start_embeddings`` and ``forward_end_embeddings``
    for the forward direction of the bidirectional representation.
    See above for a full description.
backward_combination : ``str``, optional (default = "x-y").
    The method used to combine the ``backward_start_embeddings`` and ``backward_end_embeddings``
    for the backward direction of the bidirectional representation.
    See above for a full description.
num_width_embeddings : ``int``, optional (default = None).
    Specifies the number of buckets to use when representing
    span width features.
span_width_embedding_dim : ``int``, optional (default = None).
    The embedding size for the span_width features.
bucket_widths : ``bool``, optional (default = False).
    Whether to bucket the span widths into log-space buckets. If ``False``,
    the raw span widths are used.
use_sentinels : ``bool``, optional (default = ``True``).
    If ``True``, sentinels are used to represent exclusive span indices for the elements
    in the first and last positions in the sequence (as the exclusive indices for these
    elements are outside of the the sequence boundary). This is not strictly necessary,
    as you may know that your exclusive start and end indices are always within your sequence
    representation, such as if you have appended/prepended <START> and <END> tokens to your
    sequence.

### forward
```python
BidirectionalEndpointSpanExtractor.forward(self, sequence_tensor:torch.FloatTensor, span_indices:torch.LongTensor, sequence_mask:torch.LongTensor=None, span_indices_mask:torch.LongTensor=None) -> torch.FloatTensor
```

Given a sequence tensor, extract spans and return representations of
them. Span representation can be computed in many different ways,
such as concatenation of the start and end spans, attention over the
vectors contained inside the span, etc.

Parameters
----------
sequence_tensor : ``torch.FloatTensor``, required.
    A tensor of shape (batch_size, sequence_length, embedding_size)
    representing an embedded sequence of words.
span_indices : ``torch.LongTensor``, required.
    A tensor of shape ``(batch_size, num_spans, 2)``, where the last
    dimension represents the inclusive start and end indices of the
    span to be extracted from the ``sequence_tensor``.
sequence_mask : ``torch.LongTensor``, optional (default = ``None``).
    A tensor of shape (batch_size, sequence_length) representing padded
    elements of the sequence.
span_indices_mask : ``torch.LongTensor``, optional (default = ``None``).
    A tensor of shape (batch_size, num_spans) representing the valid
    spans in the ``indices`` tensor. This mask is optional because
    sometimes it's easier to worry about masking after calling this
    function, rather than passing a mask directly.

Returns
-------
A tensor of shape ``(batch_size, num_spans, embedded_span_size)``,
where ``embedded_span_size`` depends on the way spans are represented.

