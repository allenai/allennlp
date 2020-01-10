# allennlp.models.coreference_resolution.coref

## CoreferenceResolver
```python
CoreferenceResolver(self, vocab:allennlp.data.vocabulary.Vocabulary, text_field_embedder:allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder, context_layer:allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder, mention_feedforward:allennlp.modules.feedforward.FeedForward, antecedent_feedforward:allennlp.modules.feedforward.FeedForward, feature_size:int, max_span_width:int, spans_per_word:float, max_antecedents:int, lexical_dropout:float=0.2, initializer:allennlp.nn.initializers.InitializerApplicator=<allennlp.nn.initializers.InitializerApplicator object at 0x13a040e80>, regularizer:Union[allennlp.nn.regularizers.regularizer_applicator.RegularizerApplicator, NoneType]=None) -> None
```

This ``Model`` implements the coreference resolution model described "End-to-end Neural
Coreference Resolution"
<https://www.semanticscholar.org/paper/End-to-end-Neural-Coreference-Resolution-Lee-He/3f2114893dc44eacac951f148fbff142ca200e83>
by Lee et al., 2017.
The basic outline of this model is to get an embedded representation of each span in the
document. These span representations are scored and used to prune away spans that are unlikely
to occur in a coreference cluster. For the remaining spans, the model decides which antecedent
span (if any) they are coreferent with. The resulting coreference links, after applying
transitivity, imply a clustering of the spans in the document.

Parameters
----------
vocab : ``Vocabulary``
text_field_embedder : ``TextFieldEmbedder``
    Used to embed the ``text`` ``TextField`` we get as input to the model.
context_layer : ``Seq2SeqEncoder``
    This layer incorporates contextual information for each word in the document.
mention_feedforward : ``FeedForward``
    This feedforward network is applied to the span representations which is then scored
    by a linear layer.
antecedent_feedforward : ``FeedForward``
    This feedforward network is applied to pairs of span representation, along with any
    pairwise features, which is then scored by a linear layer.
feature_size : ``int``
    The embedding size for all the embedded features, such as distances or span widths.
max_span_width : ``int``
    The maximum width of candidate spans.
spans_per_word: float, required.
    A multiplier between zero and one which controls what percentage of candidate mention
    spans we retain with respect to the number of words in the document.
max_antecedents: int, required.
    For each mention which survives the pruning stage, we consider this many antecedents.
lexical_dropout : ``int``
    The probability of dropping out dimensions of the embedded text.
initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
    Used to initialize the model parameters.
regularizer : ``RegularizerApplicator``, optional (default=``None``)
    If provided, will be used to calculate the regularization penalty during training.

### forward
```python
CoreferenceResolver.forward(self, text:Dict[str, torch.LongTensor], spans:torch.IntTensor, span_labels:torch.IntTensor=None, metadata:List[Dict[str, Any]]=None) -> Dict[str, torch.Tensor]
```

Parameters
----------
text : ``Dict[str, torch.LongTensor]``, required.
    The output of a ``TextField`` representing the text of
    the document.
spans : ``torch.IntTensor``, required.
    A tensor of shape (batch_size, num_spans, 2), representing the inclusive start and end
    indices of candidate spans for mentions. Comes from a ``ListField[SpanField]`` of
    indices into the text of the document.
span_labels : ``torch.IntTensor``, optional (default = None).
    A tensor of shape (batch_size, num_spans), representing the cluster ids
    of each span, or -1 for those which do not appear in any clusters.
metadata : ``List[Dict[str, Any]]``, optional (default = None).
    A metadata dictionary for each instance in the batch. We use the "original_text" and "clusters" keys
    from this dictionary, which respectively have the original text and the annotated gold coreference
    clusters for that instance.

Returns
-------
An output dictionary consisting of:
top_spans : ``torch.IntTensor``
    A tensor of shape ``(batch_size, num_spans_to_keep, 2)`` representing
    the start and end word indices of the top spans that survived the pruning stage.
antecedent_indices : ``torch.IntTensor``
    A tensor of shape ``(num_spans_to_keep, max_antecedents)`` representing for each top span
    the index (with respect to top_spans) of the possible antecedents the model considered.
predicted_antecedents : ``torch.IntTensor``
    A tensor of shape ``(batch_size, num_spans_to_keep)`` representing, for each top span, the
    index (with respect to antecedent_indices) of the most likely antecedent. -1 means there
    was no predicted link.
loss : ``torch.FloatTensor``, optional
    A scalar loss to be optimised.

### decode
```python
CoreferenceResolver.decode(self, output_dict:Dict[str, torch.Tensor])
```

Converts the list of spans and predicted antecedent indices into clusters
of spans for each element in the batch.

Parameters
----------
output_dict : ``Dict[str, torch.Tensor]``, required.
    The result of calling :func:`forward` on an instance or batch of instances.

Returns
-------
The same output dictionary, but with an additional ``clusters`` key:

clusters : ``List[List[List[Tuple[int, int]]]]``
    A nested list, representing, for each instance in the batch, the list of clusters,
    which are in turn comprised of a list of (start, end) inclusive spans into the
    original document.

### get_metrics
```python
CoreferenceResolver.get_metrics(self, reset:bool=False) -> Dict[str, float]
```

Returns a dictionary of metrics. This method will be called by
:class:`allennlp.training.Trainer` in order to compute and use model metrics for early
stopping and model serialization.  We return an empty dictionary here rather than raising
as it is not required to implement metrics for a new model.  A boolean `reset` parameter is
passed, as frequently a metric accumulator will have some state which should be reset
between epochs. This is also compatible with :class:`~allennlp.training.Metric`s. Metrics
should be populated during the call to ``forward``, with the
:class:`~allennlp.training.Metric` handling the accumulation of the metric until this
method is called.

