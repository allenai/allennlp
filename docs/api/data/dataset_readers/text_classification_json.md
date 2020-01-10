# allennlp.data.dataset_readers.text_classification_json

## TextClassificationJsonReader
```python
TextClassificationJsonReader(self, token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]=None, tokenizer:allennlp.data.tokenizers.tokenizer.Tokenizer=None, segment_sentences:bool=False, max_sequence_length:int=None, skip_label_indexing:bool=False, lazy:bool=False) -> None
```

Reads tokens and their labels from a labeled text classification dataset.
Expects a "text" field and a "label" field in JSON format.

The output of ``read`` is a list of ``Instance`` s with the fields:
    tokens : ``TextField`` and
    label : ``LabelField``

Parameters
----------
token_indexers : ``Dict[str, TokenIndexer]``, optional
    optional (default=``{"tokens": SingleIdTokenIndexer()}``)
    We use this to define the input representation for the text.
    See :class:`TokenIndexer`.
tokenizer : ``Tokenizer``, optional (default = ``{"tokens": SpacyTokenizer()}``)
    Tokenizer to use to split the input text into words or other kinds of tokens.
segment_sentences : ``bool``, optional (default = ``False``)
    If True, we will first segment the text into sentences using SpaCy and then tokenize words.
    Necessary for some models that require pre-segmentation of sentences, like the Hierarchical
    Attention Network (https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf).
max_sequence_length : ``int``, optional (default = ``None``)
    If specified, will truncate tokens to specified maximum length.
skip_label_indexing : ``bool``, optional (default = ``False``)
    Whether or not to skip label indexing. You might want to skip label indexing if your
    labels are numbers, so the dataset reader doesn't re-number them starting from 0.
lazy : ``bool``, optional, (default = ``False``)
    Whether or not instances can be read lazily.

### text_to_instance
```python
TextClassificationJsonReader.text_to_instance(self, text:str, label:Union[str, int]=None) -> allennlp.data.instance.Instance
```

Parameters
----------
text : ``str``, required.
    The text to classify
label : ``str``, optional, (default = None).
    The label for this text.

Returns
-------
An ``Instance`` containing the following fields:
    tokens : ``TextField``
        The tokens in the sentence or phrase.
    label : ``LabelField``
        The label label of the sentence or phrase.

