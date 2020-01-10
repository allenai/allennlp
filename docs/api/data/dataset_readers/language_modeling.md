# allennlp.data.dataset_readers.language_modeling

## LanguageModelingReader
```python
LanguageModelingReader(self, tokens_per_instance:int=None, tokenizer:allennlp.data.tokenizers.tokenizer.Tokenizer=None, token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]=None, lazy:bool=False) -> None
```

Reads a text file and converts it into a ``Dataset`` suitable for training a language model.

Note that there's one issue that needs to be fixed before this is actually usable for language
modeling - the way start and end tokens for sentences are handled is not correct; we need to
add a sentence splitter before this will be done right.

Parameters
----------
tokens_per_instance : ``int``, optional (default=``None``)
    If this is ``None``, we will have each training instance be a single sentence.  If this is
    not ``None``, we will instead take all sentences, including their start and stop tokens,
    line them up, and split the tokens into groups of this number, for more efficient training
    of the language model.
tokenizer : ``Tokenizer``, optional (default=``SpacyTokenizer()``)
    We use this ``Tokenizer`` for the text.  See :class:`Tokenizer`.
token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
    We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    Note that the `output` representation will always be single token IDs - if you've specified
    a ``SingleIdTokenIndexer`` here, we use the first one you specify.  Otherwise, we create
    one with default parameters.

### text_to_instance
```python
LanguageModelingReader.text_to_instance(self, sentence:str) -> allennlp.data.instance.Instance
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

