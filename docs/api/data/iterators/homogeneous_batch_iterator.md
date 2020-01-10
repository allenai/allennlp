# allennlp.data.iterators.homogeneous_batch_iterator

## HomogeneousBatchIterator
```python
HomogeneousBatchIterator(self, batch_size:int=32, instances_per_epoch:int=None, max_instances_in_memory:int=None, cache_instances:bool=False, track_epoch:bool=False, partition_key:str='dataset', skip_smaller_batches:bool=False) -> None
```

This iterator takes a dataset of potentially heterogeneous instances
and yields back homogeneous batches. It assumes that each instance has
some ``MetadataField`` indicating what "type" of instance it is
and bases its notion of homogeneity on that (and, in particular, not on
inspecting the "field signature" of the instance.)

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
partition_key : ``str``, optional, (default = "dataset")
    The key of the ``MetadataField`` indicating what "type" of instance this is.
skip_smaller_batches : bool, optional, (default = False)
    When the number of data samples is not dividable by `batch_size`,
    some batches might be smaller than `batch_size`.
    If set to `True`, those smaller batches will be discarded.

