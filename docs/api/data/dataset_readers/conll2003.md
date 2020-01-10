# allennlp.data.dataset_readers.conll2003

## Conll2003DatasetReader
```python
Conll2003DatasetReader(self, token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]=None, tag_label:str='ner', feature_labels:Sequence[str]=(), lazy:bool=False, coding_scheme:str='IOB1', label_namespace:str='labels') -> None
```

Reads instances from a pretokenised file where each line is in the following format:

WORD POS-TAG CHUNK-TAG NER-TAG

with a blank line indicating the end of each sentence
and '-DOCSTART- -X- -X- O' indicating the end of each article,
and converts it into a ``Dataset`` suitable for sequence tagging.

Each ``Instance`` contains the words in the ``"tokens"`` ``TextField``.
The values corresponding to the ``tag_label``
values will get loaded into the ``"tags"`` ``SequenceLabelField``.
And if you specify any ``feature_labels`` (you probably shouldn't),
the corresponding values will get loaded into their own ``SequenceLabelField`` s.

This dataset reader ignores the "article" divisions and simply treats
each sentence as an independent ``Instance``. (Technically the reader splits sentences
on any combination of blank lines and "DOCSTART" tags; in particular, it does the right
thing on well formed inputs.)

Parameters
----------
token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
    We use this to define the input representation for the text.  See :class:`TokenIndexer`.
tag_label : ``str``, optional (default=``ner``)
    Specify `ner`, `pos`, or `chunk` to have that tag loaded into the instance field `tag`.
feature_labels : ``Sequence[str]``, optional (default=``()``)
    These labels will be loaded as features into the corresponding instance fields:
    ``pos`` -> ``pos_tags``, ``chunk`` -> ``chunk_tags``, ``ner`` -> ``ner_tags``
    Each will have its own namespace : ``pos_tags``, ``chunk_tags``, ``ner_tags``.
    If you want to use one of the tags as a `feature` in your model, it should be
    specified here.
coding_scheme : ``str``, optional (default=``IOB1``)
    Specifies the coding scheme for ``ner_labels`` and ``chunk_labels``.
    Valid options are ``IOB1`` and ``BIOUL``.  The ``IOB1`` default maintains
    the original IOB1 scheme in the CoNLL 2003 NER data.
    In the IOB1 scheme, I is a token inside a span, O is a token outside
    a span and B is the beginning of span immediately following another
    span of the same type.
label_namespace : ``str``, optional (default=``labels``)
    Specifies the namespace for the chosen ``tag_label``.

### text_to_instance
```python
Conll2003DatasetReader.text_to_instance(self, tokens:List[allennlp.data.tokenizers.token.Token], pos_tags:List[str]=None, chunk_tags:List[str]=None, ner_tags:List[str]=None) -> allennlp.data.instance.Instance
```

We take `pre-tokenized` input here, because we don't have a tokenizer in this class.

