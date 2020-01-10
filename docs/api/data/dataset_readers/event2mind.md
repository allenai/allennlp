# allennlp.data.dataset_readers.event2mind

## Event2MindDatasetReader
```python
Event2MindDatasetReader(self, source_tokenizer:allennlp.data.tokenizers.tokenizer.Tokenizer=None, target_tokenizer:allennlp.data.tokenizers.tokenizer.Tokenizer=None, source_token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]=None, target_token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]=None, source_add_start_token:bool=True, dummy_instances_for_vocab_generation:bool=False, lazy:bool=False) -> None
```

Reads instances from the Event2Mind dataset.

This dataset is CSV and has the columns:
Source,Event,Xintent,Xemotion,Otheremotion,Xsent,Osent

Source is the provenance of the given instance. Event is free-form English
text. The Xintent, Xemotion, and Otheremotion columns are JSON arrays
containing the intention of "person x", the reaction to the event by
"person x" and the reaction to the event by others. The remaining columns
are not used.

For instance:
rocstory,PersonX talks to PersonX's mother,"[""to keep in touch""]","[""accomplished""]","[""loved""]",5.0,5.0

Currently we only consume the event, intent and emotions, not the sentiments.

`START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.

Parameters
----------
source_tokenizer : ``Tokenizer``, optional
    Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
    to ``SpacyTokenizer()``.
target_tokenizer : ``Tokenizer``, optional
    Tokenizer to use to split the output sequences (during training) into words or other kinds
    of tokens. Defaults to ``source_tokenizer``.
source_token_indexers : ``Dict[str, TokenIndexer]``, optional
    Indexers used to define input (source side) token representations. Defaults to
    ``{"tokens": SingleIdTokenIndexer()}``.
target_token_indexers : ``Dict[str, TokenIndexer]``, optional
    Indexers used to define output (target side) token representations. Defaults to
    ``source_token_indexers``.
source_add_start_token : ``bool``, (optional, default=True)
    Whether or not to add ``START_SYMBOL`` to the beginning of the source sequence.
dummy_instances_for_vocab_generation : ``bool`` (optional, default=False)
    Whether to generate instances that use each token of input precisely
    once. Normally we instead generate all combinations of Source, Xintent,
    Xemotion and Otheremotion columns which distorts the underlying token
    counts. This flag should be used exclusively with the ``dry-run``
    command as the instances generated will be nonsensical outside the
    context of vocabulary generation.

### text_to_instance
```python
Event2MindDatasetReader.text_to_instance(self, source_string:str, xintent_string:str=None, xreact_string:str=None, oreact_string:str=None) -> allennlp.data.instance.Instance
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

