# allennlp.data.dataset_readers.semantic_dependency_parsing

## parse_sentence
```python
parse_sentence(sentence_blob:str) -> Tuple[List[Dict[str, str]], List[Tuple[int, int]], List[str]]
```

Parses a chunk of text in the SemEval SDP format.

Each word in the sentence is returned as a dictionary with the following
format:
'id': '1',
'form': 'Pierre',
'lemma': 'Pierre',
'pos': 'NNP',
'head': '2',   # Note that this is the `syntactic` head.
'deprel': 'nn',
'top': '-',
'pred': '+',
'frame': 'named:x-c'

Along with a list of arcs and their corresponding tags. Note that
in semantic dependency parsing words can have more than one head
(it is not a tree), meaning that the list of arcs and tags are
not tied to the length of the sentence.

## SemanticDependenciesDatasetReader
```python
SemanticDependenciesDatasetReader(self, token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]=None, lazy:bool=False) -> None
```

Reads a file in the SemEval 2015 Task 18 (Broad-coverage Semantic Dependency Parsing)
format.

Parameters
----------
token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
    The token indexers to be applied to the words TextField.

### text_to_instance
```python
SemanticDependenciesDatasetReader.text_to_instance(self, tokens:List[str], pos_tags:List[str]=None, arc_indices:List[Tuple[int, int]]=None, arc_tags:List[str]=None) -> allennlp.data.instance.Instance
```

Does whatever tokenization or processing is necessary to go from textual input to an
``Instance``.  The primary intended use for this is with a
:class:`~allennlp.predictors.predictor.Predictor`, which gets text input as a JSON
object and needs to process it to be input to a model.

The intent here is to share code between :func:`_read` and what happens at
model serving time, or any other time you want to make a prediction from new data.  We need
to process the data in the same way it was done at training time.  Allowing the
``DatasetReader`` to process new text lets us accomplish this, as we can just call
``DatasetReader.text_to_instance`` when serving predictions.

The input type here is rather vaguely specified, unfortunately.  The ``Predictor`` will
have to make some assumptions about the kind of ``DatasetReader`` that it's using, in order
to pass it the right information.

