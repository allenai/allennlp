# allennlp.data.dataset_readers.coreference_resolution.conll

## canonicalize_clusters
```python
canonicalize_clusters(clusters:DefaultDict[int, List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]
```

The CONLL 2012 data includes 2 annotated spans which are identical,
but have different ids. This checks all clusters for spans which are
identical, and if it finds any, merges the clusters containing the
identical spans.

## ConllCorefReader
```python
ConllCorefReader(self, max_span_width:int, token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]=None, lazy:bool=False) -> None
```

Reads a single CoNLL-formatted file. This is the same file format as used in the
:class:`~allennlp.data.dataset_readers.semantic_role_labelling.SrlReader`, but is preprocessed
to dump all documents into a single file per train, dev and test split. See
scripts/compile_coref_data.sh for more details of how to pre-process the Ontonotes 5.0 data
into the correct format.

Returns a ``Dataset`` where the ``Instances`` have four fields : ``text``, a ``TextField``
containing the full document text, ``spans``, a ``ListField[SpanField]`` of inclusive start and
end indices for span candidates, and ``metadata``, a ``MetadataField`` that stores the instance's
original text. For data with gold cluster labels, we also include the original ``clusters``
(a list of list of index pairs) and a ``SequenceLabelField`` of cluster ids for every span
candidate.

Parameters
----------
max_span_width : ``int``, required.
    The maximum width of candidate spans to consider.
token_indexers : ``Dict[str, TokenIndexer]``, optional
    This is used to index the words in the document.  See :class:`TokenIndexer`.
    Default is ``{"tokens": SingleIdTokenIndexer()}``.

### text_to_instance
```python
ConllCorefReader.text_to_instance(self, sentences:List[List[str]], gold_clusters:Union[List[List[Tuple[int, int]]], NoneType]=None) -> allennlp.data.instance.Instance
```

Parameters
----------
sentences : ``List[List[str]]``, required.
    A list of lists representing the tokenised words and sentences in the document.
gold_clusters : ``Optional[List[List[Tuple[int, int]]]]``, optional (default = None)
    A list of all clusters in the document, represented as word spans. Each cluster
    contains some number of spans, which can be nested and overlap, but will never
    exactly match between clusters.

Returns
-------
An ``Instance`` containing the following ``Fields``:
    text : ``TextField``
        The text of the full document.
    spans : ``ListField[SpanField]``
        A ListField containing the spans represented as ``SpanFields``
        with respect to the document text.
    span_labels : ``SequenceLabelField``, optional
        The id of the cluster which each possible span belongs to, or -1 if it does
         not belong to a cluster. As these labels have variable length (it depends on
         how many spans we are considering), we represent this a as a ``SequenceLabelField``
         with respect to the ``spans ``ListField``.

