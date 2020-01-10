# allennlp.models.crf_tagger

## CrfTagger
```python
CrfTagger(self, vocab:allennlp.data.vocabulary.Vocabulary, text_field_embedder:allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder, encoder:allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder, label_namespace:str='labels', feedforward:Union[allennlp.modules.feedforward.FeedForward, NoneType]=None, label_encoding:Union[str, NoneType]=None, include_start_end_transitions:bool=True, constrain_crf_decoding:bool=None, calculate_span_f1:bool=None, dropout:Union[float, NoneType]=None, verbose_metrics:bool=False, initializer:allennlp.nn.initializers.InitializerApplicator=<allennlp.nn.initializers.InitializerApplicator object at 0x139c18978>, regularizer:Union[allennlp.nn.regularizers.regularizer_applicator.RegularizerApplicator, NoneType]=None, top_k:int=1) -> None
```

The ``CrfTagger`` encodes a sequence of text with a ``Seq2SeqEncoder``,
then uses a Conditional Random Field model to predict a tag for each token in the sequence.

Parameters
----------
vocab : ``Vocabulary``, required
    A Vocabulary, required in order to compute sizes for input/output projections.
text_field_embedder : ``TextFieldEmbedder``, required
    Used to embed the tokens ``TextField`` we get as input to the model.
encoder : ``Seq2SeqEncoder``
    The encoder that we will use in between embedding tokens and predicting output tags.
label_namespace : ``str``, optional (default=``labels``)
    This is needed to compute the SpanBasedF1Measure metric.
    Unless you did something unusual, the default value should be what you want.
feedforward : ``FeedForward``, optional, (default = None).
    An optional feedforward layer to apply after the encoder.
label_encoding : ``str``, optional (default=``None``)
    Label encoding to use when calculating span f1 and constraining
    the CRF at decoding time . Valid options are "BIO", "BIOUL", "IOB1", "BMES".
    Required if ``calculate_span_f1`` or ``constrain_crf_decoding`` is true.
include_start_end_transitions : ``bool``, optional (default=``True``)
    Whether to include start and end transition parameters in the CRF.
constrain_crf_decoding : ``bool``, optional (default=``None``)
    If ``True``, the CRF is constrained at decoding time to
    produce valid sequences of tags. If this is ``True``, then
    ``label_encoding`` is required. If ``None`` and
    label_encoding is specified, this is set to ``True``.
    If ``None`` and label_encoding is not specified, it defaults
    to ``False``.
calculate_span_f1 : ``bool``, optional (default=``None``)
    Calculate span-level F1 metrics during training. If this is ``True``, then
    ``label_encoding`` is required. If ``None`` and
    label_encoding is specified, this is set to ``True``.
    If ``None`` and label_encoding is not specified, it defaults
    to ``False``.
dropout:  ``float``, optional (default=``None``)
verbose_metrics : ``bool``, optional (default = False)
    If true, metrics will be returned per label class in addition
    to the overall statistics.
initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
    Used to initialize the model parameters.
regularizer : ``RegularizerApplicator``, optional (default=``None``)
    If provided, will be used to calculate the regularization penalty during training.
top_k : ``int``, optional (default=``1``)
    If provided, the number of parses to return from the crf in output_dict['top_k_tags'].
    Top k parses are returned as a list of dicts, where each dictionary is of the form:
    {"tags": List, "score": float}.
    The "tags" value for the first dict in the list for each data_item will be the top
    choice, and will equal the corresponding item in output_dict['tags']

### forward
```python
CrfTagger.forward(self, tokens:Dict[str, torch.LongTensor], tags:torch.LongTensor=None, metadata:List[Dict[str, Any]]=None, **kwargs) -> Dict[str, torch.Tensor]
```

Parameters
----------
tokens : ``Dict[str, torch.LongTensor]``, required
    The output of ``TextField.as_array()``, which should typically be passed directly to a
    ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
    tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is : ``{"tokens":
    Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
    for the ``TokenIndexers`` when you created the ``TextField`` representing your
    sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
    which knows how to combine different word representations into a single vector per
    token in your input.
tags : ``torch.LongTensor``, optional (default = ``None``)
    A torch tensor representing the sequence of integer gold class labels of shape
    ``(batch_size, num_tokens)``.
metadata : ``List[Dict[str, Any]]``, optional, (default = None)
    metadata containg the original words in the sentence to be tagged under a 'words' key.

Returns
-------
An output dictionary consisting of:

logits : ``torch.FloatTensor``
    The logits that are the output of the ``tag_projection_layer``
mask : ``torch.LongTensor``
    The text field mask for the input tokens
tags : ``List[List[int]]``
    The predicted tags using the Viterbi algorithm.
loss : ``torch.FloatTensor``, optional
    A scalar loss to be optimised. Only computed if gold label ``tags`` are provided.

### decode
```python
CrfTagger.decode(self, output_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]
```

Converts the tag ids to the actual tags.
``output_dict["tags"]`` is a list of lists of tag_ids,
so we use an ugly nested list comprehension.

### get_metrics
```python
CrfTagger.get_metrics(self, reset:bool=False) -> Dict[str, float]
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

