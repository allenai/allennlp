# allennlp.data.tokenizers.spacy_tokenizer

## SpacyTokenizer
```python
SpacyTokenizer(self, language:str='en_core_web_sm', pos_tags:bool=False, parse:bool=False, ner:bool=False, keep_spacy_tokens:bool=False, split_on_spaces:bool=False, start_tokens:Union[List[str], NoneType]=None, end_tokens:Union[List[str], NoneType]=None) -> None
```

A ``Tokenizer`` that uses spaCy's tokenizer.  It's fast and reasonable - this is the
recommended ``Tokenizer``. By default it will return allennlp Tokens,
which are small, efficient NamedTuples (and are serializable). If you want
to keep the original spaCy tokens, pass keep_spacy_tokens=True.  Note that we leave one particular piece of
post-processing for later: the decision of whether or not to lowercase the token.  This is for
two reasons: (1) if you want to make two different casing decisions for whatever reason, you
won't have to run the tokenizer twice, and more importantly (2) if you want to lowercase words
for your word embedding, but retain capitalization in a character-level representation, we need
to retain the capitalization here.

Parameters
----------
language : ``str``, optional, (default="en_core_web_sm")
    Spacy model name.
pos_tags : ``bool``, optional, (default=False)
    If ``True``, performs POS tagging with spacy model on the tokens.
    Generally used in conjunction with :class:`~allennlp.data.token_indexers.pos_tag_indexer.PosTagIndexer`.
parse : ``bool``, optional, (default=False)
    If ``True``, performs dependency parsing with spacy model on the tokens.
    Generally used in conjunction with :class:`~allennlp.data.token_indexers.pos_tag_indexer.DepLabelIndexer`.
ner : ``bool``, optional, (default=False)
    If ``True``, performs dependency parsing with spacy model on the tokens.
    Generally used in conjunction with :class:`~allennlp.data.token_indexers.ner_tag_indexer.NerTagIndexer`.
keep_spacy_tokens : ``bool``, optional, (default=False)
    If ``True``, will preserve spacy token objects, We copy spacy tokens into our own class by default instead
    because spacy Cython Tokens can't be pickled.
split_on_spaces : ``bool``, optional, (default=False)
    If ``True``, will split by spaces without performing tokenization.
    Used when your data is already tokenized, but you want to perform pos, ner or parsing on the tokens.
start_tokens : ``Optional[List[str]]``, optional, (default=None)
    If given, these tokens will be added to the beginning of every string we tokenize.
end_tokens : ``Optional[List[str]]``, optional, (default=None)
    If given, these tokens will be added to the end of every string we tokenize.

### batch_tokenize
```python
SpacyTokenizer.batch_tokenize(self, texts:List[str]) -> List[List[allennlp.data.tokenizers.token.Token]]
```

Batches together tokenization of several texts, in case that is faster for particular
tokenizers.

By default we just do this without batching.  Override this in your tokenizer if you have a
good way of doing batched computation.

### tokenize
```python
SpacyTokenizer.tokenize(self, text:str) -> List[allennlp.data.tokenizers.token.Token]
```

Actually implements splitting words into tokens.

Returns
-------
tokens : ``List[Token]``

