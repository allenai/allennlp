# allennlp.data.dataset_readers.next_token_lm

## NextTokenLmReader
```python
NextTokenLmReader(self, tokenizer:allennlp.data.tokenizers.tokenizer.Tokenizer=None, token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]=None, lazy:bool=False) -> None
```

Creates ``Instances`` suitable for use in predicting a single next token using a language
model.  The :class:`Field` s that we create are the following: an input ``TextField`` and a
target token ``TextField`` (we only ver have a single token, but we use a ``TextField`` so we
can index it the same way as our input, typically with a single
``PretrainedTransformerIndexer``).

NOTE: This is not fully functional!  It was written to put together a demo for interpreting and
attacking language models, not for actually training anything.  It would be a really bad idea
to use this setup for training language models, as it would be incredibly inefficient.  The
only purpose of this class is for a demo.

Parameters
----------
tokenizer : ``Tokenizer``, optional (default=``WhitespaceTokenizer()``)
    We use this ``Tokenizer`` for the text.  See :class:`Tokenizer`.
token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
    We use this to define the input representation for the text, and to get ids for the mask
    targets.  See :class:`TokenIndexer`.

### text_to_instance
```python
NextTokenLmReader.text_to_instance(self, sentence:str=None, tokens:List[allennlp.data.tokenizers.token.Token]=None, target:str=None) -> allennlp.data.instance.Instance
```

Does whatever tokenization or processing is necessary to go from textual input to an
``Instance``.  The primary intended use for this is with a
:class:`~allennlp.predictors.predictor.Predictor`, which gets text input as a JSON
object and needs to process it to be input to a model.

The intent here is to share code between :func:`_read` and what happens at
model serving time, or any other time you want to make a prediction from new data.  We need
to process the data in the same way it was done at training time.  Allowing the
``DatasetReader`` to process new text lets us accomplish this, as we can just call
``DatasetReader.text_to_instance`` when serving predictions.

The input type here is rather vaguely specified, unfortunately.  The ``Predictor`` will
have to make some assumptions about the kind of ``DatasetReader`` that it's using, in order
to pass it the right information.

