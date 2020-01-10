# allennlp.data.dataset_readers.semantic_role_labeling

## SrlReader
```python
SrlReader(self, token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]=None, domain_identifier:str=None, lazy:bool=False, bert_model_name:str=None) -> None
```

This DatasetReader is designed to read in the English OntoNotes v5.0 data
for semantic role labelling. It returns a dataset of instances with the
following fields:

tokens : ``TextField``
    The tokens in the sentence.
verb_indicator : ``SequenceLabelField``
    A sequence of binary indicators for whether the word is the verb for this frame.
tags : ``SequenceLabelField``
    A sequence of Propbank tags for the given verb in a BIO format.

Parameters
----------
token_indexers : ``Dict[str, TokenIndexer]``, optional
    We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    Default is ``{"tokens": SingleIdTokenIndexer()}``.
domain_identifier : ``str``, (default = None)
    A string denoting a sub-domain of the Ontonotes 5.0 dataset to use. If present, only
    conll files under paths containing this domain identifier will be processed.
bert_model_name : ``Optional[str]``, (default = None)
    The BERT model to be wrapped. If you specify a bert_model here, then we will
    assume you want to use BERT throughout; we will use the bert tokenizer,
    and will expand your tags and verb indicators accordingly. If not,
    the tokens will be indexed as normal with the token_indexers.

Returns
-------
A ``Dataset`` of ``Instances`` for Semantic Role Labelling.

### text_to_instance
```python
SrlReader.text_to_instance(self, tokens:List[allennlp.data.tokenizers.token.Token], verb_label:List[int], tags:List[str]=None) -> allennlp.data.instance.Instance
```

We take `pre-tokenized` input here, along with a verb label.  The verb label should be a
one-hot binary vector, the same length as the tokens, indicating the position of the verb
to find arguments for.

