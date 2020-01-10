# allennlp.data.dataset_readers.snli

## SnliReader
```python
SnliReader(self, tokenizer:allennlp.data.tokenizers.tokenizer.Tokenizer=None, token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]=None, lazy:bool=False) -> None
```

Reads a file from the Stanford Natural Language Inference (SNLI) dataset.  This data is
formatted as jsonl, one json-formatted instance per line.  The keys in the data are
"gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "label",
"premise" and "hypothesis", along with a metadata field containing the tokenized strings of the
premise and hypothesis.

Parameters
----------
tokenizer : ``Tokenizer``, optional (default=``SpacyTokenizer()``)
    We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
    We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.

### text_to_instance
```python
SnliReader.text_to_instance(self, premise:str, hypothesis:str, label:str=None) -> allennlp.data.instance.Instance
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

