# allennlp.data.tokenizers.tokenizer

## Tokenizer
```python
Tokenizer(self, /, *args, **kwargs)
```

A ``Tokenizer`` splits strings of text into tokens.  Typically, this either splits text into
word tokens or character tokens, and those are the two tokenizer subclasses we have implemented
here, though you could imagine wanting to do other kinds of tokenization for structured or
other inputs.

See the parameters to, e.g., :class:`~.SpacyTokenizer`, or whichever tokenizer
you want to use.

If the base input to your model is words, you should use a :class:`~.SpacyTokenizer`, even if
you also want to have a character-level encoder to get an additional vector for each word
token.  Splitting word tokens into character arrays is handled separately, in the
:class:`..token_representations.TokenRepresentation` class.

### default_implementation
str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.
### batch_tokenize
```python
Tokenizer.batch_tokenize(self, texts:List[str]) -> List[List[allennlp.data.tokenizers.token.Token]]
```

Batches together tokenization of several texts, in case that is faster for particular
tokenizers.

By default we just do this without batching.  Override this in your tokenizer if you have a
good way of doing batched computation.

### tokenize
```python
Tokenizer.tokenize(self, text:str) -> List[allennlp.data.tokenizers.token.Token]
```

Actually implements splitting words into tokens.

Returns
-------
tokens : ``List[Token]``

