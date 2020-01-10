# allennlp.models.masked_language_model

## MaskedLanguageModel
```python
MaskedLanguageModel(self, vocab:allennlp.data.vocabulary.Vocabulary, text_field_embedder:allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder, language_model_head:allennlp.modules.language_model_heads.language_model_head.LanguageModelHead, contextualizer:allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder=None, target_namespace:str='bert', dropout:float=0.0, initializer:allennlp.nn.initializers.InitializerApplicator=None) -> None
```

The ``MaskedLanguageModel`` embeds some input tokens (including some which are masked),
contextualizes them, then predicts targets for the masked tokens, computing a loss against
known targets.

NOTE: This was developed for use in a demo, not for training.  It's possible that it will still
work for training a masked LM, but it is very likely that some other code would be much more
efficient for that.  This `does` compute correct gradients of the loss, because we use that in
our demo, so in principle it should be able to train a model, we just don't necessarily endorse
that use.

Parameters
----------
vocab : ``Vocabulary``
text_field_embedder : ``TextFieldEmbedder``
    Used to embed the indexed tokens we get in ``forward``.
language_model_head : ``LanguageModelHead``
    The ``torch.nn.Module`` that goes from the hidden states output by the contextualizer to
    logits over some output vocabulary.
contextualizer : ``Seq2SeqEncoder``, optional (default=None)
    Used to "contextualize" the embeddings.  This is optional because the contextualization
    might actually be done in the text field embedder.
target_namespace : ``str``, optional (default='bert')
    Namespace to use to convert predicted token ids to strings in ``Model.decode``.
dropout : ``float``, optional (default=0.0)
    If specified, dropout is applied to the contextualized embeddings before computation of
    the softmax. The contextualized embeddings themselves are returned without dropout.

### forward
```python
MaskedLanguageModel.forward(self, tokens:Dict[str, torch.LongTensor], mask_positions:torch.LongTensor, target_ids:Dict[str, torch.LongTensor]=None) -> Dict[str, torch.Tensor]
```

Parameters
----------
tokens : ``Dict[str, torch.LongTensor]``
    The output of ``TextField.as_tensor()`` for a batch of sentences.
mask_positions : ``torch.LongTensor``
    The positions in ``tokens`` that correspond to [MASK] tokens that we should try to fill
    in.  Shape should be (batch_size, num_masks).
target_ids : ``Dict[str, torch.LongTensor]``
    This is a list of token ids that correspond to the mask positions we're trying to fill.
    It is the output of a ``TextField``, purely for convenience, so we can handle wordpiece
    tokenizers and such without having to do crazy things in the dataset reader.  We assume
    that there is exactly one entry in the dictionary, and that it has a shape identical to
    ``mask_positions`` - one target token per mask position.

### decode
```python
MaskedLanguageModel.decode(self, output_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]
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

