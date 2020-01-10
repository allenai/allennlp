# allennlp.data.iterators.same_language_iterator

## SameLanguageIterator
```python
SameLanguageIterator(self, sorting_keys:List[Tuple[str, str]], padding_noise:float=0.1, biggest_batch_first:bool=False, batch_size:int=32, instances_per_epoch:int=None, max_instances_in_memory:int=None, cache_instances:bool=False, track_epoch:bool=False, maximum_samples_per_batch:Tuple[str, int]=None, skip_smaller_batches:bool=False) -> None
```

Splits batches into batches containing the same language.
The language of each instance is determined by looking at the 'lang' value
in the metadata.

It takes the same parameters as :class:`allennlp.data.iterators.BucketIterator`

