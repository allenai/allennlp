# allennlp.data.iterators.multiprocess_iterator

## MultiprocessIterator
```python
MultiprocessIterator(self, base_iterator:allennlp.data.iterators.data_iterator.DataIterator, num_workers:int=1, output_queue_size:int=1000) -> None
```

Wraps another ```DataIterator``` and uses it to generate tensor dicts
using multiple processes.

Parameters
----------
base_iterator : ``DataIterator``
    The ``DataIterator`` for generating tensor dicts. It will be shared among
    processes, so it should not be stateful in any way.
num_workers : ``int``, optional (default = 1)
    The number of processes used for generating tensor dicts.
output_queue_size : ``int``, optional (default = 1000)
    The size of the output queue on which tensor dicts are placed to be consumed.
    You might need to increase this if you're generating tensor dicts too quickly.

