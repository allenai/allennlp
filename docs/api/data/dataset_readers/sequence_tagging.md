# allennlp.data.dataset_readers.sequence_tagging

## SequenceTaggingDatasetReader
```python
SequenceTaggingDatasetReader(self, word_tag_delimiter:str='###', token_delimiter:str=None, token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]=None, lazy:bool=False) -> None
```

Reads instances from a pretokenised file where each line is in the following format:

WORD###TAG [TAB] WORD###TAG [TAB] .....


and converts it into a ``Dataset`` suitable for sequence tagging. You can also specify
alternative delimiters in the constructor.

Parameters
----------
word_tag_delimiter: ``str``, optional (default=``"###"``)
    The text that separates each WORD from its TAG.
token_delimiter: ``str``, optional (default=``None``)
    The text that separates each WORD-TAG pair from the next pair. If ``None``
    then the line will just be split on whitespace.
token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
    We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    Note that the `output` tags will always correspond to single token IDs based on how they
    are pre-tokenised in the data file.

### text_to_instance
```python
SequenceTaggingDatasetReader.text_to_instance(self, tokens:List[allennlp.data.tokenizers.token.Token], tags:List[str]=None) -> allennlp.data.instance.Instance
```

We take `pre-tokenized` input here, because we don't have a tokenizer in this class.

