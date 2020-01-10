# allennlp.data.dataset_readers.universal_dependencies

## UniversalDependenciesDatasetReader
```python
UniversalDependenciesDatasetReader(self, token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]=None, use_language_specific_pos:bool=False, tokenizer:allennlp.data.tokenizers.tokenizer.Tokenizer=None, lazy:bool=False) -> None
```

Reads a file in the conllu Universal Dependencies format.

Parameters
----------
token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
    The token indexers to be applied to the words TextField.
use_language_specific_pos : ``bool``, optional (default = False)
    Whether to use UD POS tags, or to use the language specific POS tags
    provided in the conllu format.
tokenizer : ``Tokenizer``, optional, default = None
    A tokenizer to use to split the text. This is useful when the tokens that you pass
    into the model need to have some particular attribute. Typically it is not necessary.

### text_to_instance
```python
UniversalDependenciesDatasetReader.text_to_instance(self, words:List[str], upos_tags:List[str], dependencies:List[Tuple[str, int]]=None) -> allennlp.data.instance.Instance
```

Parameters
----------
words : ``List[str]``, required.
    The words in the sentence to be encoded.
upos_tags : ``List[str]``, required.
    The universal dependencies POS tags for each word.
dependencies : ``List[Tuple[str, int]]``, optional (default = None)
    A list of  (head tag, head index) tuples. Indices are 1 indexed,
    meaning an index of 0 corresponds to that word being the root of
    the dependency tree.

Returns
-------
An instance containing words, upos tags, dependency head tags and head
indices as fields.

