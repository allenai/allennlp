# allennlp.data.iterators.pass_through_iterator

## PassThroughIterator
```python
PassThroughIterator(self)
```

An iterator which performs no batching or shuffling of instances, only tensorization. E.g,
instances are effectively passed 'straight through' the iterator.

This is essentially the same as a BasicIterator with shuffling disabled, the batch size set
to 1, and maximum samples per batch disabled. The only difference is that this iterator
removes the batch dimension. This can be useful for rare situations where batching is best
performed within the dataset reader (e.g. for contiguous language modeling, or for other
problems where state is shared across batches).

