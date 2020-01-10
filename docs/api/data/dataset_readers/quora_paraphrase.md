# allennlp.data.dataset_readers.quora_paraphrase

## QuoraParaphraseDatasetReader
```python
QuoraParaphraseDatasetReader(self, lazy:bool=False, tokenizer:allennlp.data.tokenizers.tokenizer.Tokenizer=None, token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]=None) -> None
```

Reads a file from the Quora Paraphrase dataset. The train/validation/test split of the data
comes from the paper `Bilateral Multi-Perspective Matching for Natural Language Sentences
<https://arxiv.org/abs/1702.03814>`_ by Zhiguo Wang et al., 2017. Each file of the data
is a tsv file without header. The columns are is_duplicate, question1, question2, and id.
All questions are pre-tokenized and tokens are space separated. We convert these keys into
fields named "label", "premise" and "hypothesis", so that it is compatible to some existing
natural language inference algorithms.

Parameters
----------
lazy : ``bool`` (optional, default=False)
    Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
    take longer per batch.  This also allows training with datasets that are too large to fit
    in memory.
tokenizer : ``Tokenizer``, optional
    Tokenizer to use to split the premise and hypothesis into words or other kinds of tokens.
    Defaults to ``WhitespaceTokenizer``.
token_indexers : ``Dict[str, TokenIndexer]``, optional
    Indexers used to define input token representations. Defaults to ``{"tokens":
    SingleIdTokenIndexer()}``.

### text_to_instance
```python
QuoraParaphraseDatasetReader.text_to_instance(self, premise:str, hypothesis:str, label:str=None) -> allennlp.data.instance.Instance
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

