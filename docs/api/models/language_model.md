# allennlp.models.language_model

## LanguageModel
```python
LanguageModel(self, vocab:allennlp.data.vocabulary.Vocabulary, text_field_embedder:allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder, contextualizer:allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder, dropout:float=None, num_samples:int=None, sparse_embeddings:bool=False, bidirectional:bool=False, initializer:allennlp.nn.initializers.InitializerApplicator=None, regularizer:Union[allennlp.nn.regularizers.regularizer_applicator.RegularizerApplicator, NoneType]=None) -> None
```

The ``LanguageModel`` applies a "contextualizing"
``Seq2SeqEncoder`` to uncontextualized embeddings, using a ``SoftmaxLoss``
module (defined above) to compute the language modeling loss.

If bidirectional is True,  the language model is trained to predict the next and
previous tokens for each token in the input. In this case, the contextualizer must
be bidirectional. If bidirectional is False, the language model is trained to only
predict the next token for each token in the input; the contextualizer should also
be unidirectional.

If your language model is bidirectional, it is IMPORTANT that your bidirectional
``Seq2SeqEncoder`` contextualizer does not do any "peeking ahead". That is, for its
forward direction it should only consider embeddings at previous timesteps, and for
its backward direction only embeddings at subsequent timesteps. Similarly, if your
language model is unidirectional, the unidirectional contextualizer should only
consider embeddings at previous timesteps. If this condition is not met, your
language model is cheating.

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
bidirectional : ``bool``, optional (default: False)
    Train a bidirectional language model, where the contextualizer
    is used to predict the next and previous token for each input token.
    This must match the bidirectionality of the contextualizer.
regularizer : ``RegularizerApplicator``, optional (default=``None``)
    If provided, will be used to calculate the regularization penalty during training.

### delete_softmax
```python
LanguageModel.delete_softmax(self) -> None
```

Remove the softmax weights. Useful for saving memory when calculating the loss
is not necessary, e.g. in an embedder.

### num_layers
```python
LanguageModel.num_layers(self) -> int
```

Returns the depth of this LM. That is, how many layers the contextualizer has plus one for
the non-contextual layer.

### forward
```python
LanguageModel.forward(self, source:Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]
```

Computes the averaged forward (and backward, if language model is bidirectional)
LM loss from the batch.

Parameters
----------
source : ``Dict[str, torch.LongTensor]``, required.
    The output of ``Batch.as_tensor_dict()`` for a batch of sentences. By convention,
    it's required to have at least a ``"tokens"`` entry that's the output of a
    ``SingleIdTokenIndexer``, which is used to compute the language model targets.

Returns
-------
Dict with keys:

``'loss'`` : ``torch.Tensor``
    forward negative log likelihood, or the average of forward/backward
    if language model is bidirectional
``'forward_loss'`` : ``torch.Tensor``
    forward direction negative log likelihood
``'backward_loss'`` : ``torch.Tensor`` or ``None``
    backward direction negative log likelihood. If language model is not
    bidirectional, this is ``None``.
``'lm_embeddings'`` : ``Union[torch.Tensor, List[torch.Tensor]]``
    (batch_size, timesteps, embed_dim) tensor of top layer contextual representations or
    list of all layers. No dropout applied.
``'noncontextual_token_embeddings'`` : ``torch.Tensor``
    (batch_size, timesteps, token_embed_dim) tensor of bottom layer noncontextual
    representations
``'mask'`` : ``torch.Tensor``
    (batch_size, timesteps) mask for the embeddings

