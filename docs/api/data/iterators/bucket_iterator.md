# allennlp.data.iterators.bucket_iterator

## sort_by_padding
```python
sort_by_padding(instances:List[allennlp.data.instance.Instance], sorting_keys:List[Tuple[str, str]], vocab:allennlp.data.vocabulary.Vocabulary, padding_noise:float=0.0) -> List[allennlp.data.instance.Instance]
```

Sorts the instances by their padding lengths, using the keys in
``sorting_keys`` (in the order in which they are provided).  ``sorting_keys`` is a list of
``(field_name, padding_key)`` tuples.

## BucketIterator
```python
BucketIterator(self, sorting_keys:List[Tuple[str, str]], padding_noise:float=0.1, biggest_batch_first:bool=False, batch_size:int=32, instances_per_epoch:int=None, max_instances_in_memory:int=None, cache_instances:bool=False, track_epoch:bool=False, maximum_samples_per_batch:Tuple[str, int]=None, skip_smaller_batches:bool=False) -> None
```

An iterator which by default, pads batches with respect to the maximum input lengths `per
batch`. Additionally, you can provide a list of field names and padding keys which the dataset
will be sorted by before doing this batching, causing inputs with similar length to be batched
together, making computation more efficient (as less time is wasted on padded elements of the
batch).

Parameters
----------
sorting_keys : List[Tuple[str, str]]
    To bucket inputs into batches, we want to group the instances by padding length, so that we
    minimize the amount of padding necessary per batch. In order to do this, we need to know
    which fields need what type of padding, and in what order.

    For example, ``[("sentence1", "num_tokens"), ("sentence2", "num_tokens"), ("sentence1",
    "num_token_characters")]`` would sort a dataset first by the "num_tokens" of the
    "sentence1" field, then by the "num_tokens" of the "sentence2" field, and finally by the
    "num_token_characters" of the "sentence1" field.  TODO(mattg): we should have some
    documentation somewhere that gives the standard padding keys used by different fields.
padding_noise : float, optional (default=.1)
    When sorting by padding length, we add a bit of noise to the lengths, so that the sorting
    isn't deterministic.  This parameter determines how much noise we add, as a percentage of
    the actual padding value for each instance.
biggest_batch_first : bool, optional (default=False)
    This is largely for testing, to see how large of a batch you can safely use with your GPU.
    This will let you try out the largest batch that you have in the data `first`, so that if
    you're going to run out of memory, you know it early, instead of waiting through the whole
    epoch to find out at the end that you're going to crash.

    Note that if you specify ``max_instances_in_memory``, the first batch will only be the
    biggest from among the first "max instances in memory" instances.
batch_size : int, optional, (default = 32)
    The size of each batch of instances yielded when calling the iterator.
instances_per_epoch : int, optional, (default = None)
    See :class:`BasicIterator`.
max_instances_in_memory : int, optional, (default = None)
    See :class:`BasicIterator`.
maximum_samples_per_batch : ``Tuple[str, int]``, (default = None)
    See :class:`BasicIterator`.
skip_smaller_batches : bool, optional, (default = False)
    When the number of data samples is not dividable by `batch_size`,
    some batches might be smaller than `batch_size`.
    If set to `True`, those smaller batches will be discarded.

