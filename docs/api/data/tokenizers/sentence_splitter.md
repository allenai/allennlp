# allennlp.data.tokenizers.sentence_splitter

## SentenceSplitter
```python
SentenceSplitter(self, /, *args, **kwargs)
```

A ``SentenceSplitter`` splits strings into sentences.

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
### split_sentences
```python
SentenceSplitter.split_sentences(self, text:str) -> List[str]
```

Splits a ``text`` :class:`str` paragraph into a list of :class:`str`, where each is a sentence.

### batch_split_sentences
```python
SentenceSplitter.batch_split_sentences(self, texts:List[str]) -> List[List[str]]
```

Default implementation is to just iterate over the texts and call ``split_sentences``.

## SpacySentenceSplitter
```python
SpacySentenceSplitter(self, language:str='en_core_web_sm', rule_based:bool=False) -> None
```

A ``SentenceSplitter`` that uses spaCy's built-in sentence boundary detection.

Spacy's default sentence splitter uses a dependency parse to detect sentence boundaries, so
it is slow, but accurate.

Another option is to use rule-based sentence boundary detection. It's fast and has a small memory footprint,
since it uses punctuation to detect sentence boundaries. This can be activated with the `rule_based` flag.

By default, ``SpacySentenceSplitter`` calls the default spacy boundary detector.

### split_sentences
```python
SpacySentenceSplitter.split_sentences(self, text:str) -> List[str]
```

Splits a ``text`` :class:`str` paragraph into a list of :class:`str`, where each is a sentence.

### batch_split_sentences
```python
SpacySentenceSplitter.batch_split_sentences(self, texts:List[str]) -> List[List[str]]
```

This method lets you take advantage of spacy's batch processing.

