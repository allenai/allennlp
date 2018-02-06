# Laziness in AllenNLP

By default, a `DatasetReader` returns instances as a `list`
containing every instance in the dataset. There are several
reasons why you might not want this behavior:

* your dataset is too large to fit into memory
* you want each iteration through the dataset to do some sort of sampling
* you want to start training on your data immediately
  rather than wait for the whole dataset to be processed and indexed.
  (In this case you'd want to first use the
   [make-vocab](https://github.com/allenai/allennlp/blob/master/allennlp/commands/make_vocab.py)
   command to create your vocabulary so that training really does start immediately.)

In these cases
you'll want your `DatasetReader` to be "lazy"; that is,
to create and yield up instances as needed rather than all at once.

This tutorial will show both you how to create `DatasetReader`s
that allow this lazy behavior, and how to handle this laziness
when training a model.

If you're not interested in the details,
there's an extremely brief [How-To](../how_to/laziness.md)
that just tells you the steps to follow.
But really you should be interested in the details.

## You specify laziness in the `DatasetReader` constructor

If you look at the [constructor](https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/dataset_reader.py#L43)
for the base `DatasetReader` class, you can see that it takes a single parameter:

```python
    def __init__(self, lazy: bool = False) -> None:
        self.lazy = lazy
```

This means that if you want your `DatasetReader` subclass to allow for laziness,
its constructor needs to pass a `lazy` value to the `DatasetReader` constructor,
and its `from_params` method needs to allow for such a parameter. All of the
dataset readers included with AllenNLP are built this way.

For example, look at the
[`SnliReader`](https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/snli.py).

Its constructor takes a `lazy` parameter and passes it to the superclass constructor:

```python
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
```

And its `from_params` looks for a `lazy` configuration parameter
(defaulting to `False` if it's not specified):

```python
    @classmethod
    def from_params(cls, params: Params) -> 'SnliReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return SnliReader(tokenizer=tokenizer,
                          token_indexers=token_indexers,
                          lazy=lazy)
```

Any dataset reader that you want to handle lazy datasets should behave likewise.

## Laziness in `DatasetReader.read`

The primary public interface of the `DatasetReader` is its `read` method, which is implemented in
the base class (and which I've stripped down to its essence):

```python
    def read(self, file_path: str) -> Iterable[Instance]:
        if self.lazy:
            return _LazyInstances(lambda: iter(self._read(file_path)))
        else:
            return ensure_list(self._read(file_path))
```

In both cases, it calls the private `_read` method, which you have to implement,
and which itself returns an `Iterable[Instance]`. More on that below.

If the dataset reader was instantiated with `lazy=False`,
it just makes sure that the returned instances are in a `list` and returns that list.
That list is loaded into memory and (because it's a list) can be iterated over repeatedly
after the initial call to `.read()`.

The more interesting case is when the dataset reader was instantiated with `lazy=True`.
In that case we return a `_LazyInstances` initialized with a lambda function
that calls `_read()`. `_LazyInstances` is just a simple wrapper to produce
lazy iterables (again, I stripped it down to just its essence):

```python
class _LazyInstances(Iterable):
    def __init__(self, instance_generator: Callable[[], Iterator[Instance]]) -> None:
        self.instance_generator = instance_generator

    def __iter__(self) -> Iterator[Instance]:
        return self.instance_generator()
```

What this means is that the result is an iterable that calls the provided
function each time it's iterated over. With the lambda we passed it,
it will call `self._read(file_path)` each time it's iterated over.

In other words, if you implement your own dataset reader and do something like the following:

```python
reader = MyDatasetReader(lazy=True)
instances = reader.read('my_instances.txt')
for epoch in range(10):
    for instance in instances:
        process(instance)
```

Then each epoch's `for instance in instances` results in a *new* call
to `MyDatasetReader._read()`, and your instances will be read from disk
10 times.

## Laziness in `YourDatasetReader._read()`

When you implement a `DatasetReader` subclass, you have to override the private method

```python
    def _read(self, file_path: str) -> Iterable[Instance]:
```

with the logic to generate the all the `Instance`s in your dataset.

Your implementation could return a `list`,
but you should strongly consider implementing `_read` as a
[generator](https://docs.python.org/3/howto/functional.html#generators)
that `yield`s one instance at a time; otherwise your dataset reader
can't be used in a lazy way. (Indeed, if your `_read()` implementation
returns a `list` you'll get an error if you try to set `lazy=True`
in your dataset reader.)

This means that you should not do something like

```python
    def _read(self, file_path: str) -> Iterable[Instance]:
        """Don't write your _read function this way"""
        instances = []
        # logic to iterate over file
            # some kind of for loop
            # instance = ...
            instances.append(instance)
        return instances
```

but should instead do

```python
    def _read(self, file_path: str) -> Iterable[Instance]:
        """Do write your _read function this way"""
        # logic to iterate over file
            # some kind of for loop
            # instance = ...
            yield instance
```

In the `lazy=False` case, `DatasetReader.read()` will
materialize your instances to a list before returning them anyway.

## Laziness in `experiment.json`

If you implemented your dataset reader as specified above,
laziness just requires adding `"lazy": true` to your
dataset reader config:

```json
  "dataset_reader": {
    "type": "snli",
    "lazy": true,
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      }
    }
  },
```

## Laziness in `DataIterator`

AllenNLP uses a `DataIterator` abstraction to iterate over datasets
using configurable batching, shuffling, and so on.  The included
dataset readers all work with lazy datasets, but they have extra options
that you might want to specify in that case. In particular, we'll
look at the
[`BasicIterator`](https://github.com/allenai/allennlp/blob/master/allennlp/data/iterators/basic_iterator.py):

```python
    def __init__(self,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None) -> None:
        self._batch_size = batch_size
        self._instances_per_epoch = instances_per_epoch
        self._max_instances_in_memory = max_instances_in_memory
```

If you use the default parameters and call this iterator on a lazy dataset,
it will lazily read 32 instances at a time, package those instances into a
`Batch`, and yield them one by one. If you specify `shuffle=True` when you call
the iterator, it will shuffle the 32 instances within each batch, but will always
serve batches in the same order.

If you have a sense for how many instances can fit in memory, you can specify
the `max_instances_in_memory` parameter. For instance, if you were to set that
to 3200, the iterator would load 3200 instances at a time into memory,
potentially shuffle them, and then create 100 batches of size 32 and yield
them up before loading the next 3200 instances.

This is especially useful if you are using the `BucketIterator`,
which can order your instances by e.g. sentence length. If you don't specify
`max_instances_in_memory`, you'd just get each batch of 32 ordered by sentence length,
which isn't particularly helpful.

The other option here is `instances_per_epoch`. By default, each epoch is a single
pass through the dataset. However, if you had millions of instances,
you might want each epoch to consist of a fraction of the dataset,
in which case you can specify the value here.
