# allennlp.data.dataset_readers.babi

## BabiReader
```python
BabiReader(self, keep_sentences:bool=False, token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]=None, lazy:bool=False) -> None
```

Reads one single task in the bAbI tasks format as formulated in
Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks
(https://arxiv.org/abs/1502.05698). Since this class handle a single file,
if one wants to load multiple tasks together it has to merge them into a
single file and use this reader.

Parameters
----------
keep_sentences : ``bool``, optional, (default = ``False``)
    Whether to keep each sentence in the context or to concatenate them.
    Default is ``False`` that corresponds to concatenation.
token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
    We use this to define the input representation for the text.  See :class:`TokenIndexer`.
lazy : ``bool``, optional, (default = ``False``)
    Whether or not instances can be consumed lazily.

### text_to_instance
```python
BabiReader.text_to_instance(self, context:List[List[str]], question:List[str], answer:str, supports:List[int]) -> allennlp.data.instance.Instance
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

