# allennlp.data.dataset_readers.simple_language_modeling

## SimpleLanguageModelingDatasetReader
```python
SimpleLanguageModelingDatasetReader(self, tokenizer:allennlp.data.tokenizers.tokenizer.Tokenizer=None, token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]=None, max_sequence_length:int=None, start_tokens:List[str]=None, end_tokens:List[str]=None) -> None
```

Reads sentences, one per line, for language modeling. This does not handle arbitrarily formatted
text with sentences spanning multiple lines.

Parameters
----------
tokenizer : ``Tokenizer``, optional
    Tokenizer to use to split the input sentences into words or other kinds of tokens. Defaults
    to ``SpacyTokenizer()``.
token_indexers : ``Dict[str, TokenIndexer]``, optional
    Indexers used to define input token representations. Defaults to
    ``{"tokens": SingleIdTokenIndexer()}``.
max_sequence_length : ``int``, optional
    If specified, sentences with more than this number of tokens will be dropped.
start_tokens : ``List[str]``, optional (default=``None``)
    These are prepended to the tokens provided to the ``TextField``.
end_tokens : ``List[str]``, optional (default=``None``)
    These are appended to the tokens provided to the ``TextField``.

### text_to_instance
```python
SimpleLanguageModelingDatasetReader.text_to_instance(self, sentence:str) -> allennlp.data.instance.Instance
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

