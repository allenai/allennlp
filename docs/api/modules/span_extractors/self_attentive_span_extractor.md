# allennlp.modules.span_extractors.self_attentive_span_extractor

## SelfAttentiveSpanExtractor
```python
SelfAttentiveSpanExtractor(self, input_dim:int) -> None
```

Computes span representations by generating an unnormalized attention score for each
word in the document. Spans representations are computed with respect to these
scores by normalising the attention scores for words inside the span.

Given these attention distributions over every span, this module weights the
corresponding vector representations of the words in the span by this distribution,
returning a weighted representation of each span.

Parameters
----------
input_dim : ``int``, required.
    The final dimension of the ``sequence_tensor``.

Returns
-------
attended_text_embeddings : ``torch.FloatTensor``.
    A tensor of shape (batch_size, num_spans, input_dim), which each span representation
    is formed by locally normalising a global attention over the sequence. The only way
    in which the attention distribution differs over different spans is in the set of words
    over which they are normalized.

### forward
```python
SelfAttentiveSpanExtractor.forward(self, sequence_tensor:torch.FloatTensor, span_indices:torch.LongTensor, span_indices_mask:torch.LongTensor=None) -> torch.FloatTensor
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

