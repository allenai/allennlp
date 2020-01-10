# allennlp.data.dataset_readers.conll2000

## Conll2000DatasetReader
```python
Conll2000DatasetReader(self, token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]=None, tag_label:str='chunk', feature_labels:Sequence[str]=(), lazy:bool=False, coding_scheme:str='BIO', label_namespace:str='labels') -> None
```

Reads instances from a pretokenised file where each line is in the following format:

WORD POS-TAG CHUNK-TAG

with a blank line indicating the end of each sentence
and converts it into a ``Dataset`` suitable for sequence tagging.

Each ``Instance`` contains the words in the ``"tokens"`` ``TextField``.
The values corresponding to the ``tag_label``
values will get loaded into the ``"tags"`` ``SequenceLabelField``.
And if you specify any ``feature_labels`` (you probably shouldn't),
the corresponding values will get loaded into their own ``SequenceLabelField`` s.

Parameters
----------
token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
    We use this to define the input representation for the text.  See :class:`TokenIndexer`.
tag_label : ``str``, optional (default=``chunk``)
    Specify `pos`, or `chunk` to have that tag loaded into the instance field `tag`.
feature_labels : ``Sequence[str]``, optional (default=``()``)
    These labels will be loaded as features into the corresponding instance fields:
    ``pos`` -> ``pos_tags`` or ``chunk`` -> ``chunk_tags``.
    Each will have its own namespace : ``pos_tags`` or ``chunk_tags``.
    If you want to use one of the tags as a `feature` in your model, it should be
    specified here.
coding_scheme : ``str``, optional (default=``BIO``)
    Specifies the coding scheme for ``chunk_labels``.
    Valid options are ``BIO`` and ``BIOUL``.  The ``BIO`` default maintains
    the original BIO scheme in the CoNLL 2000 chunking data.
    In the BIO scheme, B is a token starting a span, I is a token continuing a span, and
    O is a token outside of a span.
label_namespace : ``str``, optional (default=``labels``)
    Specifies the namespace for the chosen ``tag_label``.

### text_to_instance
```python
Conll2000DatasetReader.text_to_instance(self, tokens:List[allennlp.data.tokenizers.token.Token], pos_tags:List[str]=None, chunk_tags:List[str]=None) -> allennlp.data.instance.Instance
```

We take `pre-tokenized` input here, because we don't have a tokenizer in this class.

