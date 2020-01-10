# allennlp.data.dataset_readers.universal_dependencies_multilang

## get_file_paths
```python
get_file_paths(pathname:str, languages:List[str])
```

Gets a list of all files by the pathname with the given language ids.
Filenames are assumed to have the language identifier followed by a dash
as a prefix (e.g. en-universal.conll).

Parameters
----------
pathname :  ``str``, required.
    An absolute or relative pathname (can contain shell-style wildcards)
languages : ``List[str]``, required
    The language identifiers to use.

Returns
-------
A list of tuples (language id, file path).

## UniversalDependenciesMultiLangDatasetReader
```python
UniversalDependenciesMultiLangDatasetReader(self, languages:List[str], token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]=None, use_language_specific_pos:bool=False, lazy:bool=False, alternate:bool=True, is_first_pass_for_vocab:bool=True, instances_per_file:int=32) -> None
```

Reads multiple files in the conllu Universal Dependencies format.
All files should be in the same directory and the filenames should have
the language identifier followed by a dash as a prefix (e.g. en-universal.conll)
When using the alternate option, the reader alternates randomly between
the files every instances_per_file. The is_first_pass_for_vocab disables
this behaviour for the first pass (could be useful for a single full path
over the dataset in order to generate a vocabulary).

Notice: when using the alternate option, one should also use the ``instances_per_epoch``
option for the iterator. Otherwise, each epoch will loop infinitely.

Parameters
----------
languages : ``List[str]``, required
    The language identifiers to use.
token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
    The token indexers to be applied to the words TextField.
use_language_specific_pos : ``bool``, optional (default = False)
    Whether to use UD POS tags, or to use the language specific POS tags
    provided in the conllu format.
alternate : ``bool``, optional (default = True)
    Whether to alternate between input files.
is_first_pass_for_vocab : ``bool``, optional (default = True)
    Whether the first pass will be for generating the vocab. If true,
    the first pass will run over the entire dataset of each file (even if alternate is on).
instances_per_file : ``int``, optional (default = 32)
    The amount of consecutive cases to sample from each input file when alternating.

### text_to_instance
```python
UniversalDependenciesMultiLangDatasetReader.text_to_instance(self, lang:str, words:List[str], upos_tags:List[str], dependencies:List[Tuple[str, int]]=None) -> allennlp.data.instance.Instance
```

Parameters
----------
lang : ``str``, required.
    The language identifier.
words : ``List[str]``, required.
    The words in the sentence to be encoded.
upos_tags : ``List[str]``, required.
    The universal dependencies POS tags for each word.
dependencies ``List[Tuple[str, int]]``, optional (default = None)
    A list of  (head tag, head index) tuples. Indices are 1 indexed,
    meaning an index of 0 corresponds to that word being the root of
    the dependency tree.

Returns
-------
An instance containing words, upos tags, dependency head tags and head
indices as fields. The language identifier is stored in the metadata.

