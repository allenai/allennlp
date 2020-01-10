# allennlp.data.tokenizers.character_tokenizer

## CharacterTokenizer
```python
CharacterTokenizer(self, byte_encoding:str=None, lowercase_characters:bool=False, start_tokens:List[str]=None, end_tokens:List[str]=None) -> None
```

A ``CharacterTokenizer`` splits strings into character tokens.

Parameters
----------
byte_encoding : str, optional (default=``None``)
    If not ``None``, we will use this encoding to encode the string as bytes, and use the byte
    sequence as characters, instead of the unicode characters in the python string.  E.g., the
    character 'á' would be a single token if this option is ``None``, but it would be two
    tokens if this option is set to ``"utf-8"``.

    If this is not ``None``, ``tokenize`` will return a ``List[int]`` instead of a
    ``List[str]``, and we will bypass the vocabulary in the ``TokenIndexer``.
lowercase_characters : ``bool``, optional (default=``False``)
    If ``True``, we will lowercase all of the characters in the text before doing any other
    operation.  You probably do not want to do this, as character vocabularies are generally
    not very large to begin with, but it's an option if you really want it.
start_tokens : ``List[str]``, optional
    If given, these tokens will be added to the beginning of every string we tokenize.  If
    using byte encoding, this should actually be a ``List[int]``, not a ``List[str]``.
end_tokens : ``List[str]``, optional
    If given, these tokens will be added to the end of every string we tokenize.  If using byte
    encoding, this should actually be a ``List[int]``, not a ``List[str]``.

### tokenize
```python
CharacterTokenizer.tokenize(self, text:str) -> List[allennlp.data.tokenizers.token.Token]
```

Actually implements splitting words into tokens.

Returns
-------
tokens : ``List[Token]``

