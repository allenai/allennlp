# allennlp.data.dataset_readers.coreference_resolution.winobias

## WinobiasReader
```python
WinobiasReader(self, max_span_width:int, token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]=None, lazy:bool=False) -> None
```

TODO(Mark): Add paper reference.

Winobias is a dataset to analyse the issue of gender bias in co-reference
resolution. It contains simple sentences with pro/anti stereotypical gender
associations with which to measure the bias of a coreference system trained
on another corpus. It is effectively a toy dataset and as such, uses very
simplistic language; it has little use outside of evaluating a model for bias.

The dataset is formatted with a single sentence per line, with a maximum of 2
non-nested coreference clusters annotated using either square or round brackets.
For example:

[The salesperson] sold (some books) to the librarian because [she] was trying to sell (them).


Returns a list of ``Instances`` which have four fields : ``text``, a ``TextField``
containing the full sentence text, ``spans``, a ``ListField[SpanField]`` of inclusive start and
end indices for span candidates, and ``metadata``, a ``MetadataField`` that stores the instance's
original text. For data with gold cluster labels, we also include the original ``clusters``
(a list of list of index pairs) and a ``SequenceLabelField`` of cluster ids for every span
candidate in the ``metadata`` also.

Parameters
----------
max_span_width : ``int``, required.
    The maximum width of candidate spans to consider.
token_indexers : ``Dict[str, TokenIndexer]``, optional
    This is used to index the words in the sentence.  See :class:`TokenIndexer`.
    Default is ``{"tokens": SingleIdTokenIndexer()}``.

### text_to_instance
```python
WinobiasReader.text_to_instance(self, sentence:List[allennlp.data.tokenizers.token.Token], gold_clusters:Union[List[List[Tuple[int, int]]], NoneType]=None) -> allennlp.data.instance.Instance
```

Parameters
----------
sentence : ``List[Token]``, required.
    The already tokenised sentence to analyse.
gold_clusters : ``Optional[List[List[Tuple[int, int]]]]``, optional (default = None)
    A list of all clusters in the sentence, represented as word spans. Each cluster
    contains some number of spans, which can be nested and overlap, but will never
    exactly match between clusters.

Returns
-------
An ``Instance`` containing the following ``Fields``:
    text : ``TextField``
        The text of the full sentence.
    spans : ``ListField[SpanField]``
        A ListField containing the spans represented as ``SpanFields``
        with respect to the sentence text.
    span_labels : ``SequenceLabelField``, optional
        The id of the cluster which each possible span belongs to, or -1 if it does
         not belong to a cluster. As these labels have variable length (it depends on
         how many spans we are considering), we represent this a as a ``SequenceLabelField``
         with respect to the ``spans ``ListField``.

