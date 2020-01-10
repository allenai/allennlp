# allennlp.data.dataset_readers.seq2seq

## Seq2SeqDatasetReader
```python
Seq2SeqDatasetReader(self, source_tokenizer:allennlp.data.tokenizers.tokenizer.Tokenizer=None, target_tokenizer:allennlp.data.tokenizers.tokenizer.Tokenizer=None, source_token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]=None, target_token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]=None, source_add_start_token:bool=True, source_add_end_token:bool=True, delimiter:str='\t', source_max_tokens:Union[int, NoneType]=None, target_max_tokens:Union[int, NoneType]=None, lazy:bool=False) -> None
```

Read a tsv file containing paired sequences, and create a dataset suitable for a
``ComposedSeq2Seq`` model, or any model with a matching API.

Expected format for each input line: <source_sequence_string>	<target_sequence_string>

The output of ``read`` is a list of ``Instance`` s with the fields:
    source_tokens : ``TextField`` and
    target_tokens : ``TextField``

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
source_add_start_token : bool, (optional, default=True)
    Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
source_add_end_token : bool, (optional, default=True)
    Whether or not to add `END_SYMBOL` to the end of the source sequence.
delimiter : str, (optional, default="	")
    Set delimiter for tsv/csv file.

### text_to_instance
```python
Seq2SeqDatasetReader.text_to_instance(self, source_string:str, target_string:str=None) -> allennlp.data.instance.Instance
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

