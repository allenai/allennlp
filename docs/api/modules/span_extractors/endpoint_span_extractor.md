# allennlp.modules.span_extractors.endpoint_span_extractor

## EndpointSpanExtractor
```python
EndpointSpanExtractor(self, input_dim:int, combination:str='x,y', num_width_embeddings:int=None, span_width_embedding_dim:int=None, bucket_widths:bool=False, use_exclusive_start_indices:bool=False) -> None
```

Represents spans as a combination of the embeddings of their endpoints. Additionally,
the width of the spans can be embedded and concatenated on to the final combination.

The following types of representation are supported, assuming that
``x = span_start_embeddings`` and ``y = span_end_embeddings``.

``x``, ``y``, ``x*y``, ``x+y``, ``x-y``, ``x/y``, where each of those binary operations
is performed elementwise.  You can list as many combinations as you want, comma separated.
For example, you might give ``x,y,x*y`` as the ``combination`` parameter to this class.
The computed similarity function would then be ``[x; y; x*y]``, which can then be optionally
concatenated with an embedded representation of the width of the span.

Parameters
----------
input_dim : ``int``, required.
    The final dimension of the ``sequence_tensor``.
combination : ``str``, optional (default = "x,y").
    The method used to combine the ``start_embedding`` and ``end_embedding``
    representations. See above for a full description.
num_width_embeddings : ``int``, optional (default = None).
    Specifies the number of buckets to use when representing
    span width features.
span_width_embedding_dim : ``int``, optional (default = None).
    The embedding size for the span_width features.
bucket_widths : ``bool``, optional (default = False).
    Whether to bucket the span widths into log-space buckets. If ``False``,
    the raw span widths are used.
use_exclusive_start_indices : ``bool``, optional (default = ``False``).
    If ``True``, the start indices extracted are converted to exclusive indices. Sentinels
    are used to represent exclusive span indices for the elements in the first
    position in the sequence (as the exclusive indices for these elements are outside
    of the the sequence boundary) so that start indices can be exclusive.
    NOTE: This option can be helpful to avoid the pathological case in which you
    want span differences for length 1 spans - if you use inclusive indices, you
    will end up with an ``x - x`` operation for length 1 spans, which is not good.

### forward
```python
EndpointSpanExtractor.forward(self, sequence_tensor:torch.FloatTensor, span_indices:torch.LongTensor, sequence_mask:torch.LongTensor=None, span_indices_mask:torch.LongTensor=None) -> None
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

