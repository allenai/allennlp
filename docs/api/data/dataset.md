# allennlp.data.dataset

A :class:`~Batch` represents a collection of ``Instance`` s to be fed
through a model.

## Batch
```python
Batch(self, instances:Iterable[allennlp.data.instance.Instance]) -> None
```

A batch of Instances. In addition to containing the instances themselves,
it contains helper functions for converting the data into tensors.

### get_padding_lengths
```python
Batch.get_padding_lengths(self) -> Dict[str, Dict[str, int]]
```

Gets the maximum padding lengths from all ``Instances`` in this batch.  Each ``Instance``
has multiple ``Fields``, and each ``Field`` could have multiple things that need padding.
We look at all fields in all instances, and find the max values for each (field_name,
padding_key) pair, returning them in a dictionary.

This can then be used to convert this batch into arrays of consistent length, or to set
model parameters, etc.

### as_tensor_dict
```python
Batch.as_tensor_dict(self, padding_lengths:Dict[str, Dict[str, int]]=None, verbose:bool=False) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
```

This method converts this ``Batch`` into a set of pytorch Tensors that can be passed
through a model.  In order for the tensors to be valid tensors, all ``Instances`` in this
batch need to be padded to the same lengths wherever padding is necessary, so we do that
first, then we combine all of the tensors for each field in each instance into a set of
batched tensors for each field.

Parameters
----------
padding_lengths : ``Dict[str, Dict[str, int]]``
    If a key is present in this dictionary with a non-``None`` value, we will pad to that
    length instead of the length calculated from the data.  This lets you, e.g., set a
    maximum value for sentence length if you want to throw out long sequences.

    Entries in this dictionary are keyed first by field name (e.g., "question"), then by
    padding key (e.g., "num_tokens").
verbose : ``bool``, optional (default=``False``)
    Should we output logging information when we're doing this padding?  If the batch is
    large, this is nice to have, because padding a large batch could take a long time.
    But if you're doing this inside of a data generator, having all of this output per
    batch is a bit obnoxious (and really slow).

Returns
-------
tensors : ``Dict[str, DataArray]``
    A dictionary of tensors, keyed by field name, suitable for passing as input to a model.
    This is a `batch` of instances, so, e.g., if the instances have a "question" field and
    an "answer" field, the "question" fields for all of the instances will be grouped
    together into a single tensor, and the "answer" fields for all instances will be
    similarly grouped in a parallel set of tensors, for batched computation. Additionally,
    for complex ``Fields``, the value of the dictionary key is not necessarily a single
    tensor.  For example, with the ``TextField``, the output is a dictionary mapping
    ``TokenIndexer`` keys to tensors. The number of elements in this sub-dictionary
    therefore corresponds to the number of ``TokenIndexers`` used to index the
    ``TextField``.  Each ``Field`` class is responsible for batching its own output.

