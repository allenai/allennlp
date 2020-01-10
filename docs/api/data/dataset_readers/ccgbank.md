# allennlp.data.dataset_readers.ccgbank

## CcgBankDatasetReader
```python
CcgBankDatasetReader(self, token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]=None, tag_label:str='ccg', feature_labels:Sequence[str]=(), label_namespace:str='labels', lazy:bool=False) -> None
```

Reads data in the "machine-readable derivation" format of the CCGbank dataset.
(see https://catalog.ldc.upenn.edu/docs/LDC2005T13/CCGbankManual.pdf, section D.2)

In particular, it pulls out the leaf nodes, which are represented as

    (<L ccg_category modified_pos original_pos token predicate_arg_category>)

The tarballed version of the dataset contains many files worth of this data,
in files /data/AUTO/xx/wsj_xxxx.auto

This dataset reader expects a single text file. Accordingly, if you're using that dataset,
you'll need to first concatenate some of those files into a training set, a validation set,
and a test set.

Parameters
----------
token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
    We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    Note that the `output` tags will always correspond to single token IDs based on how they
    are pre-tokenised in the data file.
lazy : ``bool``, optional, (default = ``False``)
    Whether or not instances can be consumed lazily.
tag_label : ``str``, optional (default=``ccg``)
    Specify ``ccg``, ``modified_pos``, ``original_pos``, or ``predicate_arg`` to
    have that tag loaded into the instance field ``tag``.
feature_labels : ``Sequence[str]``, optional (default=``()``)
    These labels will be loaded as features into the corresponding instance fields:
    ``ccg`` -> ``ccg_tags``, ``modified_pos`` -> ``modified_pos_tags``,
    ``original_pos`` -> ``original_pos_tags``, or ``predicate_arg`` -> ``predicate_arg_tags``
    Each will have its own namespace : ``ccg_tags``, ``modified_pos_tags``,
    ``original_pos_tags``, ``predicate_arg_tags``. If you want to use one of the tags
    as a feature in your model, it should be specified here.
label_namespace : ``str``, optional (default=``labels``)
    Specifies the namespace for the chosen ``tag_label``.

### text_to_instance
```python
CcgBankDatasetReader.text_to_instance(self, tokens:List[str], ccg_categories:List[str]=None, original_pos_tags:List[str]=None, modified_pos_tags:List[str]=None, predicate_arg_categories:List[str]=None) -> allennlp.data.instance.Instance
```

We take `pre-tokenized` input here, because we don't have a tokenizer in this class.

Parameters
----------
tokens : ``List[str]``, required.
    The tokens in a given sentence.
ccg_categories : ``List[str]``, optional, (default = None).
    The CCG categories for the words in the sentence. (e.g. N/N)
original_pos_tags : ``List[str]``, optional, (default = None).
    The tag assigned to the word in the Penn Treebank.
modified_pos_tags : ``List[str]``, optional, (default = None).
    The POS tag might have changed during the translation to CCG.
predicate_arg_categories : ``List[str]``, optional, (default = None).
    Encodes the word-word dependencies in the underlying predicate-
    argument structure.

Returns
-------
An ``Instance`` containing the following fields:
    tokens : ``TextField``
        The tokens in the sentence.
    tags : ``SequenceLabelField``
        The tags corresponding to the ``tag_label`` constructor argument.
    feature_label_tags : ``SequenceLabelField``
        Tags corresponding to each feature_label (if any) specified in the
        ``feature_labels`` constructor argument.

