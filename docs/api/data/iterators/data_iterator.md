# allennlp.data.iterators.data_iterator

## add_epoch_number
```python
add_epoch_number(batch:allennlp.data.dataset.Batch, epoch:int) -> allennlp.data.dataset.Batch
```

Add the epoch number to the batch instances as a MetadataField.

## DataIterator
```python
DataIterator(self, batch_size:int=32, instances_per_epoch:int=None, max_instances_in_memory:int=None, cache_instances:bool=False, track_epoch:bool=False, maximum_samples_per_batch:Tuple[str, int]=None) -> None
```

An abstract ``DataIterator`` class. ``DataIterators`` must override ``_create_batches()``.

Parameters
----------
batch_size : ``int``, optional, (default = 32)
    The size of each batch of instances yielded when calling the iterator.
instances_per_epoch : ``int``, optional, (default = None)
    If specified, each epoch will consist of precisely this many instances.
    If not specified, each epoch will consist of a single pass through the dataset.
max_instances_in_memory : ``int``, optional, (default = None)
    If specified, the iterator will load this many instances at a time into an
    in-memory list and then produce batches from one such list at a time. This
    could be useful if your instances are read lazily from disk.
cache_instances : ``bool``, optional, (default = False)
    If true, the iterator will cache the tensorized instances in memory.
    If false, it will do the tensorization anew each iteration.
track_epoch : ``bool``, optional, (default = False)
    If true, each instance will get a ``MetadataField`` containing the epoch number.
maximum_samples_per_batch : ``Tuple[str, int]``, (default = None)
    If specified, then is a tuple (padding_key, limit) and we will ensure
    that every batch is such that batch_size * sequence_length <= limit
    where sequence_length is given by the padding_key. This is done by
    moving excess instances to the next batch (as opposed to dividing a
    large batch evenly) and should result in a fairly tight packing.

### default_implementation
str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.
### get_num_batches
```python
DataIterator.get_num_batches(self, instances:Iterable[allennlp.data.instance.Instance]) -> int
```

Returns the number of batches that ``dataset`` will be split into; if you want to track
progress through the batch with the generator produced by ``__call__``, this could be
useful.

