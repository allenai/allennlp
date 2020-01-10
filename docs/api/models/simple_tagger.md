# allennlp.models.simple_tagger

## SimpleTagger
```python
SimpleTagger(self, vocab:allennlp.data.vocabulary.Vocabulary, text_field_embedder:allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder, encoder:allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder, calculate_span_f1:bool=None, label_encoding:Union[str, NoneType]=None, label_namespace:str='labels', verbose_metrics:bool=False, initializer:allennlp.nn.initializers.InitializerApplicator=<allennlp.nn.initializers.InitializerApplicator object at 0x13ae4f0f0>, regularizer:Union[allennlp.nn.regularizers.regularizer_applicator.RegularizerApplicator, NoneType]=None) -> None
```

This ``SimpleTagger`` simply encodes a sequence of text with a stacked ``Seq2SeqEncoder``, then
predicts a tag for each token in the sequence.

Parameters
----------
vocab : ``Vocabulary``, required
    A Vocabulary, required in order to compute sizes for input/output projections.
text_field_embedder : ``TextFieldEmbedder``, required
    Used to embed the ``tokens`` ``TextField`` we get as input to the model.
encoder : ``Seq2SeqEncoder``
    The encoder (with its own internal stacking) that we will use in between embedding tokens
    and predicting output tags.
calculate_span_f1 : ``bool``, optional (default=``None``)
    Calculate span-level F1 metrics during training. If this is ``True``, then
    ``label_encoding`` is required. If ``None`` and
    label_encoding is specified, this is set to ``True``.
    If ``None`` and label_encoding is not specified, it defaults
    to ``False``.
label_encoding : ``str``, optional (default=``None``)
    Label encoding to use when calculating span f1.
    Valid options are "BIO", "BIOUL", "IOB1", "BMES".
    Required if ``calculate_span_f1`` is true.
label_namespace : ``str``, optional (default=``labels``)
    This is needed to compute the SpanBasedF1Measure metric, if desired.
    Unless you did something unusual, the default value should be what you want.
verbose_metrics : ``bool``, optional (default = False)
    If true, metrics will be returned per label class in addition
    to the overall statistics.
initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
    Used to initialize the model parameters.
regularizer : ``RegularizerApplicator``, optional (default=``None``)
    If provided, will be used to calculate the regularization penalty during training.

### forward
```python
SimpleTagger.forward(self, tokens:Dict[str, torch.LongTensor], tags:torch.LongTensor=None, metadata:List[Dict[str, Any]]=None) -> Dict[str, torch.Tensor]
```

Parameters
----------
tokens : Dict[str, torch.LongTensor], required
    The output of ``TextField.as_array()``, which should typically be passed directly to a
    ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
    tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is : ``{"tokens":
    Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
    for the ``TokenIndexers`` when you created the ``TextField`` representing your
    sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
    which knows how to combine different word representations into a single vector per
    token in your input.
tags : torch.LongTensor, optional (default = None)
    A torch tensor representing the sequence of integer gold class labels of shape
    ``(batch_size, num_tokens)``.
metadata : ``List[Dict[str, Any]]``, optional, (default = None)
    metadata containing the original words in the sentence to be tagged under a 'words' key.

Returns
-------
An output dictionary consisting of:
logits : torch.FloatTensor
    A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
    unnormalised log probabilities of the tag classes.
class_probabilities : torch.FloatTensor
    A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
    a distribution of the tag classes per word.
loss : torch.FloatTensor, optional
    A scalar loss to be optimised.


### decode
```python
SimpleTagger.decode(self, output_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]
```

Does a simple position-wise argmax over each token, converts indices to string labels, and
adds a ``"tags"`` key to the dictionary with the result.

### get_metrics
```python
SimpleTagger.get_metrics(self, reset:bool=False) -> Dict[str, float]
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

