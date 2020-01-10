# allennlp.data.iterators.basic_iterator

## BasicIterator
```python
BasicIterator(self, batch_size:int=32, instances_per_epoch:int=None, max_instances_in_memory:int=None, cache_instances:bool=False, track_epoch:bool=False, maximum_samples_per_batch:Tuple[str, int]=None) -> None
```

A very basic iterator that takes a dataset, possibly shuffles it, and creates fixed sized batches.

It takes the same parameters as :class:`allennlp.data.iterators.DataIterator`

