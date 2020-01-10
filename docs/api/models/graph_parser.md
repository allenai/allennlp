# allennlp.models.graph_parser

## GraphParser
```python
GraphParser(self, vocab:allennlp.data.vocabulary.Vocabulary, text_field_embedder:allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder, encoder:allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder, tag_representation_dim:int, arc_representation_dim:int, tag_feedforward:allennlp.modules.feedforward.FeedForward=None, arc_feedforward:allennlp.modules.feedforward.FeedForward=None, pos_tag_embedding:allennlp.modules.token_embedders.embedding.Embedding=None, dropout:float=0.0, input_dropout:float=0.0, edge_prediction_threshold:float=0.5, initializer:allennlp.nn.initializers.InitializerApplicator=<allennlp.nn.initializers.InitializerApplicator object at 0x12f8b3b38>, regularizer:Union[allennlp.nn.regularizers.regularizer_applicator.RegularizerApplicator, NoneType]=None) -> None
```

A Parser for arbitrary graph structures.

Parameters
----------
vocab : ``Vocabulary``, required
    A Vocabulary, required in order to compute sizes for input/output projections.
text_field_embedder : ``TextFieldEmbedder``, required
    Used to embed the ``tokens`` ``TextField`` we get as input to the model.
encoder : ``Seq2SeqEncoder``
    The encoder (with its own internal stacking) that we will use to generate representations
    of tokens.
tag_representation_dim : ``int``, required.
    The dimension of the MLPs used for arc tag prediction.
arc_representation_dim : ``int``, required.
    The dimension of the MLPs used for arc prediction.
tag_feedforward : ``FeedForward``, optional, (default = None).
    The feedforward network used to produce tag representations.
    By default, a 1 layer feedforward network with an elu activation is used.
arc_feedforward : ``FeedForward``, optional, (default = None).
    The feedforward network used to produce arc representations.
    By default, a 1 layer feedforward network with an elu activation is used.
pos_tag_embedding : ``Embedding``, optional.
    Used to embed the ``pos_tags`` ``SequenceLabelField`` we get as input to the model.
dropout : ``float``, optional, (default = 0.0)
    The variational dropout applied to the output of the encoder and MLP layers.
input_dropout : ``float``, optional, (default = 0.0)
    The dropout applied to the embedded text input.
edge_prediction_threshold : ``int``, optional (default = 0.5)
    The probability at which to consider a scored edge to be 'present'
    in the decoded graph. Must be between 0 and 1.
initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
    Used to initialize the model parameters.
regularizer : ``RegularizerApplicator``, optional (default=``None``)
    If provided, will be used to calculate the regularization penalty during training.

### forward
```python
GraphParser.forward(self, tokens:Dict[str, torch.LongTensor], pos_tags:torch.LongTensor=None, metadata:List[Dict[str, Any]]=None, arc_tags:torch.LongTensor=None) -> Dict[str, torch.Tensor]
```

Parameters
----------
tokens : Dict[str, torch.LongTensor], required
    The output of ``TextField.as_array()``.
pos_tags : torch.LongTensor, optional (default = None)
    The output of a ``SequenceLabelField`` containing POS tags.
metadata : List[Dict[str, Any]], optional (default = None)
    A dictionary of metadata for each batch element which has keys:
        tokens : ``List[str]``, required.
            The original string tokens in the sentence.
arc_tags : torch.LongTensor, optional (default = None)
    A torch tensor representing the sequence of integer indices denoting the parent of every
    word in the dependency parse. Has shape ``(batch_size, sequence_length, sequence_length)``.

Returns
-------
An output dictionary.

### decode
```python
GraphParser.decode(self, output_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]
```

Takes the result of :func:`forward` and runs inference / decoding / whatever
post-processing you need to do your model.  The intent is that ``model.forward()`` should
produce potentials or probabilities, and then ``model.decode()`` can take those results and
run some kind of beam search or constrained inference or whatever is necessary.  This does
not handle all possible decoding use cases, but it at least handles simple kinds of
decoding.

This method `modifies` the input dictionary, and also `returns` the same dictionary.

By default in the base class we do nothing.  If your model has some special decoding step,
override this method.

### get_metrics
```python
GraphParser.get_metrics(self, reset:bool=False) -> Dict[str, float]
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

