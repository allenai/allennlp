# allennlp.data.tokenizers.whitespace_tokenizer

## WhitespaceTokenizer
```python
WhitespaceTokenizer(self, /, *args, **kwargs)
```

A ``Tokenizer`` that assumes you've already done your own tokenization somehow and have
separated the tokens by spaces.  We just split the input string on whitespace and return the
resulting list.

Note that we use ``text.split()``, which means that the amount of whitespace between the
tokens does not matter.  This will never result in spaces being included as tokens.

### tokenize
```python
WhitespaceTokenizer.tokenize(self, text:str) -> List[allennlp.data.tokenizers.token.Token]
```

Actually implements splitting words into tokens.

Returns
-------
tokens : ``List[Token]``

