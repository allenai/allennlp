# allennlp.modules.span_extractors.span_extractor

## SpanExtractor
```python
SpanExtractor(self)
```

Many NLP models deal with representations of spans inside a sentence.
SpanExtractors define methods for extracting and representing spans
from a sentence.

SpanExtractors take a sequence tensor of shape (batch_size, timesteps, embedding_dim)
and indices of shape (batch_size, num_spans, 2) and return a tensor of
shape (batch_size, num_spans, ...), forming some representation of the
spans.

### forward
```python
SpanExtractor.forward(self, sequence_tensor:torch.FloatTensor, span_indices:torch.LongTensor, sequence_mask:torch.LongTensor=None, span_indices_mask:torch.LongTensor=None)
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

### get_input_dim
```python
SpanExtractor.get_input_dim(self) -> int
```

Returns the expected final dimension of the ``sequence_tensor``.

### get_output_dim
```python
SpanExtractor.get_output_dim(self) -> int
```

Returns the expected final dimension of the returned span representation.

