# allennlp.models.bidirectional_lm

## BidirectionalLanguageModel
```python
BidirectionalLanguageModel(self, vocab:allennlp.data.vocabulary.Vocabulary, text_field_embedder:allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder, contextualizer:allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder, dropout:float=None, num_samples:int=None, sparse_embeddings:bool=False, initializer:allennlp.nn.initializers.InitializerApplicator=None, regularizer:Union[allennlp.nn.regularizers.regularizer_applicator.RegularizerApplicator, NoneType]=None) -> None
```

The ``BidirectionalLanguageModel`` applies a bidirectional "contextualizing"
``Seq2SeqEncoder`` to uncontextualized embeddings, using a ``SoftmaxLoss``
module (defined above) to compute the language modeling loss.

It is IMPORTANT that your bidirectional ``Seq2SeqEncoder`` does not do any
"peeking ahead". That is, for its forward direction it should only consider
embeddings at previous timesteps, and for its backward direction only embeddings
at subsequent timesteps. If this condition is not met, your language model is
cheating.

Parameters
----------
vocab : ``Vocabulary``
text_field_embedder : ``TextFieldEmbedder``
    Used to embed the indexed tokens we get in ``forward``.
contextualizer : ``Seq2SeqEncoder``
    Used to "contextualize" the embeddings. As described above,
    this encoder must not cheat by peeking ahead.
dropout : ``float``, optional (default: None)
    If specified, dropout is applied to the contextualized embeddings before computation of
    the softmax. The contextualized embeddings themselves are returned without dropout.
num_samples : ``int``, optional (default: None)
    If provided, the model will use ``SampledSoftmaxLoss``
    with the specified number of samples. Otherwise, it will use
    the full ``_SoftmaxLoss`` defined above.
sparse_embeddings : ``bool``, optional (default: False)
    Passed on to ``SampledSoftmaxLoss`` if True.
regularizer : ``RegularizerApplicator``, optional (default=``None``)
    If provided, will be used to calculate the regularization penalty during training.

