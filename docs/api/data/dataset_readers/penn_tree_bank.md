# allennlp.data.dataset_readers.penn_tree_bank

## PennTreeBankConstituencySpanDatasetReader
```python
PennTreeBankConstituencySpanDatasetReader(self, token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]=None, use_pos_tags:bool=True, convert_parentheses:bool=False, lazy:bool=False, label_namespace_prefix:str='', pos_label_namespace:str='pos') -> None
```

Reads constituency parses from the WSJ part of the Penn Tree Bank from the LDC.
This ``DatasetReader`` is designed for use with a span labelling model, so
it enumerates all possible spans in the sentence and returns them, along with gold
labels for the relevant spans present in a gold tree, if provided.

Parameters
----------
token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
    We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    Note that the `output` tags will always correspond to single token IDs based on how they
    are pre-tokenised in the data file.
use_pos_tags : ``bool``, optional, (default = ``True``)
    Whether or not the instance should contain gold POS tags
    as a field.
convert_parentheses : ``bool``, optional, (default = ``False``)
    Whether or not to convert special PTB parentheses tokens (e.g., "-LRB-")
    to the corresponding parentheses tokens (i.e., "(").
lazy : ``bool``, optional, (default = ``False``)
    Whether or not instances can be consumed lazily.
label_namespace_prefix : ``str``, optional, (default = ``""``)
    Prefix used for the label namespace.  The ``span_labels`` will use
    namespace ``label_namespace_prefix + 'labels'``, and if using POS
    tags their namespace is ``label_namespace_prefix + pos_label_namespace``.
pos_label_namespace : ``str``, optional, (default = ``"pos"``)
    The POS tag namespace is ``label_namespace_prefix + pos_label_namespace``.

### text_to_instance
```python
PennTreeBankConstituencySpanDatasetReader.text_to_instance(self, tokens:List[str], pos_tags:List[str]=None, gold_tree:nltk.tree.Tree=None) -> allennlp.data.instance.Instance
```

We take `pre-tokenized` input here, because we don't have a tokenizer in this class.

Parameters
----------
tokens : ``List[str]``, required.
    The tokens in a given sentence.
pos_tags : ``List[str]``, optional, (default = None).
    The POS tags for the words in the sentence.
gold_tree : ``Tree``, optional (default = None).
    The gold parse tree to create span labels from.

Returns
-------
An ``Instance`` containing the following fields:
    tokens : ``TextField``
        The tokens in the sentence.
    pos_tags : ``SequenceLabelField``
        The POS tags of the words in the sentence.
        Only returned if ``use_pos_tags`` is ``True``
    spans : ``ListField[SpanField]``
        A ListField containing all possible subspans of the
        sentence.
    span_labels : ``SequenceLabelField``, optional.
        The constituency tags for each of the possible spans, with
        respect to a gold parse tree. If a span is not contained
        within the tree, a span will have a ``NO-LABEL`` label.
    gold_tree : ``MetadataField(Tree)``
        The gold NLTK parse tree for use in evaluation.

