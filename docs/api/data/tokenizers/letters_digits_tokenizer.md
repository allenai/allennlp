# allennlp.data.tokenizers.letters_digits_tokenizer

## LettersDigitsTokenizer
```python
LettersDigitsTokenizer(self, /, *args, **kwargs)
```

A ``Tokenizer`` which keeps runs of (unicode) letters and runs of digits together, while
every other non-whitespace character becomes a separate word.

### tokenize
```python
LettersDigitsTokenizer.tokenize(self, text:str) -> List[allennlp.data.tokenizers.token.Token]
```

Actually implements splitting words into tokens.

Returns
-------
tokens : ``List[Token]``

