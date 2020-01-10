# allennlp.data.dataset_readers.ontonotes_ner

## OntonotesNamedEntityRecognition
```python
OntonotesNamedEntityRecognition(self, token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]=None, domain_identifier:str=None, coding_scheme:str='BIO', lazy:bool=False) -> None
```

This DatasetReader is designed to read in the English OntoNotes v5.0 data
for fine-grained named entity recognition. It returns a dataset of instances with the
following fields:

tokens : ``TextField``
    The tokens in the sentence.
tags : ``SequenceLabelField``
    A sequence of BIO tags for the NER classes.

Note that the "/pt/" directory of the Onotonotes dataset representing annotations
on the new and old testaments of the Bible are excluded, because they do not contain
NER annotations.

Parameters
----------
token_indexers : ``Dict[str, TokenIndexer]``, optional
    We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    Default is ``{"tokens": SingleIdTokenIndexer()}``.
domain_identifier : ``str``, (default = None)
    A string denoting a sub-domain of the Ontonotes 5.0 dataset to use. If present, only
    conll files under paths containing this domain identifier will be processed.
coding_scheme : ``str``, (default = None).
    The coding scheme to use for the NER labels. Valid options are "BIO" or "BIOUL".

Returns
-------
A ``Dataset`` of ``Instances`` for Fine-Grained NER.


### text_to_instance
```python
OntonotesNamedEntityRecognition.text_to_instance(self, tokens:List[allennlp.data.tokenizers.token.Token], ner_tags:List[str]=None) -> allennlp.data.instance.Instance
```

We take `pre-tokenized` input here, because we don't have a tokenizer in this class.

