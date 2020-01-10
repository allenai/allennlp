# allennlp.models.next_token_lm

## NextTokenLM
```python
NextTokenLM(self, vocab:allennlp.data.vocabulary.Vocabulary, text_field_embedder:allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder, language_model_head:allennlp.modules.language_model_heads.language_model_head.LanguageModelHead, contextualizer:allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder=None, target_namespace:str='bert', dropout:float=0.0, initializer:allennlp.nn.initializers.InitializerApplicator=None) -> None
```

The ``NextTokenLM`` embeds some input tokens, contextualizes them, then predicts the next word,
computing a loss against known target.

NOTE: This was developed for use in a demo, not for training.  You `definitely` don't want to
train a language model using this code; it would be incredibly inefficient.  This `does`
compute correct gradients of the loss, however, so you can use it for interesting visualization
of the gradients of a pretrained model, and it appears to be fast enough to sample from, at
least for one word at a time.  If you want to sample many tokens at a time, you'd want to
re-use some intermediate computation, so you would either need to modify this code or use
something else.

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

### decode
```python
NextTokenLM.decode(self, output_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]
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

