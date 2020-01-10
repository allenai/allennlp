# allennlp.data.dataset_readers.masked_language_modeling

## MaskedLanguageModelingReader
```python
MaskedLanguageModelingReader(self, tokenizer:allennlp.data.tokenizers.tokenizer.Tokenizer=None, token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]=None, lazy:bool=False) -> None
```

Reads a text file and converts it into a ``Dataset`` suitable for training a masked language
model.

The :class:`Field` s that we create are the following: an input ``TextField``, a mask position
``ListField[IndexField]``, and a target token ``TextField`` (the target tokens aren't a single
string of text, but we use a ``TextField`` so we can index the target tokens the same way as
our input, typically with a single ``PretrainedTransformerIndexer``).  The mask position and
target token lists are the same length.

NOTE: This is not fully functional!  It was written to put together a demo for interpreting and
attacking masked language modeling, not for actually training anything.  ``text_to_instance``
is functional, but ``_read`` is not.  To make this fully functional, you would want some
sampling strategies for picking the locations for [MASK] tokens, and probably a bunch of
efficiency / multi-processing stuff.

Parameters
----------
tokenizer : ``Tokenizer``, optional (default=``WhitespaceTokenizer()``)
    We use this ``Tokenizer`` for the text.  See :class:`Tokenizer`.
token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
    We use this to define the input representation for the text, and to get ids for the mask
    targets.  See :class:`TokenIndexer`.

### text_to_instance
```python
MaskedLanguageModelingReader.text_to_instance(self, sentence:str=None, tokens:List[allennlp.data.tokenizers.token.Token]=None, targets:List[str]=None) -> allennlp.data.instance.Instance
```

Parameters
----------
sentence : ``str``, optional
    A sentence containing [MASK] tokens that should be filled in by the model.  This input
    is superceded and ignored if ``tokens`` is given.
tokens : ``List[Token]``, optional
    An already-tokenized sentence containing some number of [MASK] tokens to be predicted.
targets : ``List[str]``, optional
    Contains the target tokens to be predicted.  The length of this list should be the same
    as the number of [MASK] tokens in the input.

