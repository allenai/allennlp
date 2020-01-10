# allennlp.models.encoder_decoders.composed_seq2seq

## ComposedSeq2Seq
```python
ComposedSeq2Seq(self, vocab:allennlp.data.vocabulary.Vocabulary, source_text_embedder:allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder, encoder:allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder, decoder:allennlp.modules.seq2seq_decoders.seq_decoder.SeqDecoder, tied_source_embedder_key:Union[str, NoneType]=None, initializer:allennlp.nn.initializers.InitializerApplicator=<allennlp.nn.initializers.InitializerApplicator object at 0x1375fd940>, regularizer:Union[allennlp.nn.regularizers.regularizer_applicator.RegularizerApplicator, NoneType]=None) -> None
```

This ``ComposedSeq2Seq`` class is a :class:`Model` which takes a sequence, encodes it, and then
uses the encoded representations to decode another sequence.  You can use this as the basis for
a neural machine translation system, an abstractive summarization system, or any other common
seq2seq problem.  The model here is simple, but should be a decent starting place for
implementing recent models for these tasks.

The ``ComposedSeq2Seq`` class composes separate ``Seq2SeqEncoder`` and ``SeqDecoder`` classes.
These parts are customizable and are independent from each other.

Parameters
----------
vocab : ``Vocabulary``, required
    Vocabulary containing source and target vocabularies. They may be under the same namespace
    (`tokens`) or the target tokens can have a different namespace, in which case it needs to
    be specified as `target_namespace`.
source_text_embedders : ``TextFieldEmbedder``, required
    Embedders for source side sequences
encoder : ``Seq2SeqEncoder``, required
    The encoder of the "encoder/decoder" model
decoder : ``SeqDecoder``, required
    The decoder of the "encoder/decoder" model
tied_source_embedder_key : ``str``, optional (default=``None``)
    If specified, this key is used to obtain token_embedder in `source_text_embedder` and
    the weights are shared/tied with the decoder's target embedding weights.
initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
    Used to initialize the model parameters.
regularizer : ``RegularizerApplicator``, optional (default=``None``)
    If provided, will be used to calculate the regularization penalty during training.

### forward
```python
ComposedSeq2Seq.forward(self, source_tokens:Dict[str, torch.LongTensor], target_tokens:Dict[str, torch.LongTensor]=None) -> Dict[str, torch.Tensor]
```

Make foward pass on the encoder and decoder for producing the entire target sequence.

Parameters
----------
source_tokens : ``Dict[str, torch.LongTensor]``
   The output of `TextField.as_array()` applied on the source `TextField`. This will be
   passed through a `TextFieldEmbedder` and then through an encoder.
target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
   Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
   target tokens are also represented as a `TextField`.

Returns
-------
Dict[str, torch.Tensor]
    The output tensors from the decoder.

### decode
```python
ComposedSeq2Seq.decode(self, output_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]
```

Finalize predictions.

### get_metrics
```python
ComposedSeq2Seq.get_metrics(self, reset:bool=False) -> Dict[str, float]
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

