# allennlp.data.dataset_readers.copynet_seq2seq

## CopyNetDatasetReader
```python
CopyNetDatasetReader(self, target_namespace:str, source_tokenizer:allennlp.data.tokenizers.tokenizer.Tokenizer=None, target_tokenizer:allennlp.data.tokenizers.tokenizer.Tokenizer=None, source_token_indexers:Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]=None, lazy:bool=False) -> None
```

Read a tsv file containing paired sequences, and create a dataset suitable for a
``CopyNet`` model, or any model with a matching API.

The expected format for each input line is: <source_sequence_string><tab><target_sequence_string>.
An instance produced by ``CopyNetDatasetReader`` will containing at least the following fields:

- ``source_tokens``: a ``TextField`` containing the tokenized source sentence,
   including the ``START_SYMBOL`` and ``END_SYMBOL``.
   This will result in a tensor of shape ``(batch_size, source_length)``.

- ``source_token_ids``: an ``ArrayField`` of size ``(batch_size, trimmed_source_length)``
  that contains an ID for each token in the source sentence. Tokens that
  match at the lowercase level will share the same ID. If ``target_tokens``
  is passed as well, these IDs will also correspond to the ``target_token_ids``
  field, i.e. any tokens that match at the lowercase level in both
  the source and target sentences will share the same ID. Note that these IDs
  have no correlation with the token indices from the corresponding
  vocabulary namespaces.

- ``source_to_target``: a ``NamespaceSwappingField`` that keeps track of the index
  of the target token that matches each token in the source sentence.
  When there is no matching target token, the OOV index is used.
  This will result in a tensor of shape ``(batch_size, trimmed_source_length)``.

- ``metadata``: a ``MetadataField`` which contains the source tokens and
  potentially target tokens as lists of strings.

When ``target_string`` is passed, the instance will also contain these fields:

- ``target_tokens``: a ``TextField`` containing the tokenized target sentence,
  including the ``START_SYMBOL`` and ``END_SYMBOL``. This will result in
  a tensor of shape ``(batch_size, target_length)``.

- ``target_token_ids``: an ``ArrayField`` of size ``(batch_size, target_length)``.
  This is calculated in the same way as ``source_token_ids``.

See the "Notes" section below for a description of how these fields are used.

Parameters
----------
target_namespace : ``str``, required
    The vocab namespace for the targets. This needs to be passed to the dataset reader
    in order to construct the NamespaceSwappingField.
source_tokenizer : ``Tokenizer``, optional
    Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
    to ``SpacyTokenizer()``.
target_tokenizer : ``Tokenizer``, optional
    Tokenizer to use to split the output sequences (during training) into words or other kinds
    of tokens. Defaults to ``source_tokenizer``.
source_token_indexers : ``Dict[str, TokenIndexer]``, optional
    Indexers used to define input (source side) token representations. Defaults to
    ``{"tokens": SingleIdTokenIndexer()}``.

Notes
-----
By ``source_length`` we are referring to the number of tokens in the source
sentence including the ``START_SYMBOL`` and ``END_SYMBOL``, while
``trimmed_source_length`` refers to the number of tokens in the source sentence
*excluding* the ``START_SYMBOL`` and ``END_SYMBOL``, i.e.
``trimmed_source_length = source_length - 2``.

On the other hand, ``target_length`` is the number of tokens in the target sentence
*including* the ``START_SYMBOL`` and ``END_SYMBOL``.

In the context where there is a ``batch_size`` dimension, the above refer
to the maximum of their individual values across the batch.

In regards to the fields in an ``Instance`` produced by this dataset reader,
``source_token_ids`` and ``target_token_ids`` are primarily used during training
to determine whether a target token is copied from a source token (or multiple matching
source tokens), while ``source_to_target`` is primarily used during prediction
to combine the copy scores of source tokens with the generation scores for matching
tokens in the target namespace.

### text_to_instance
```python
CopyNetDatasetReader.text_to_instance(self, source_string:str, target_string:str=None) -> allennlp.data.instance.Instance
```

Turn raw source string and target string into an ``Instance``.

Parameters
----------
source_string : ``str``, required
target_string : ``str``, optional (default = None)

Returns
-------
Instance
    See the above for a description of the fields that the instance will contain.

