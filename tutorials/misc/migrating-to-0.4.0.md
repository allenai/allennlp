`allennlp` 0.4.0 introduces several breaking changes.
A couple of them you are likely to run into,
most of them you are not.

# CHANGES THAT YOU ARE LIKELY TO RUN INTO

## `allennlp.run evaluate`

The `--archive-file` parameter for `evaluate` was changed to a positional parameter,
as it is in our other commands. This means that your previous

```bash
python -m allennlp.run evaluate --archive-file model.tar.gz --evaluation-data-file data.txt
```

will need to be changed to

```bash
python -m allennlp.run evaluate model.tar.gz --evaluation-data-file data.txt
```

## `DatasetReader`

If you have written your own `DatasetReader` subclass, it will need a few small changes.

Previously every dataset reader was *eager*;
that is, it returned a `Dataset` object that wrapped a `list` of `Instance`s
that necessarily stayed in memory throughout the training process.

In 0.4.0 we enabled *lazy* datasets, which required some minor changes in
how `DatasetReader`s work.

Previously you would have overridden the `read` function with something like

```python
def read(self, file_path: str) -> Dataset:
    instances: List[Instance] = []
    # open some data_file
    for line in data_file:
       instance = ...
       instances.append(instance)
    return Dataset(instances)
```

In 0.4.0 the `read()` function is already implemented in the base `DatasetReader`
class in a way that allows every dataset reader to generate instances either
_lazily_ (that is, as needed, and anew each iteration) or
_eagerly_ (that is, generated once and stored in a list)
in a way that's transparent to the user.

So in 0.4.0 your `DatasetReader` subclass needs to instead
override the private `_read` function
and `yield` instances one at time.

This should only be like a 2-line change:

```python
# new name _read and return type: Iterable[Instance]
def _read(self, file_path: str) -> Iterable[Instance]:
    # open some data_file
    for line in data_file:
       instance = ...
       # yield instances rather than collecting them in a list
       yield instance
```

In addition, the base `DatasetReader` constructor now takes a `lazy: bool` parameter,
which means that your subclass constructor should also take that parameter
(unless you don't want to allow laziness, but why wouldn't you?)
and explicitly pass it to the superclass constructor:

```python
class MyDatasetReader(DatasetReader)
    def __init__(self,
                 # other arguments
                 lazy: bool = False):
        super().__init__(lazy)
        # whatever other initialization you need
```

For the reasoning behind this change, see the [Laziness tutorial](https://github.com/allenai/allennlp/blob/master/tutorials/getting_started/laziness.md).

# CHANGES YOU ARE MUCH LESS LIKELY TO RUN INTO

If you only ever create `Model`s and `DatasetReader`s and
use the command line tools (`python -m allennlp.run ...`) to train and evaluate them,
you shouldn't have to change anything else. However, if you've written your own training loop,
(or if you're just really interested in how AllenNLP works), there are a few more changes you should know about.

## `Dataset`

AllenNLP 0.4.0 gets rid of the `Dataset` class / abstraction.
Previously, a `Dataset` was a wrapper for a concrete list of instances
that contained logic for _indexing_ them with some vocabulary,
and for converting them into `Tensor`s. And, as mentioned above,
previously `DatasetReader.read()` returned such a dataset.

In 0.4.0, `DatasetReader.read()` returns an `Iterable[Instance]`,
which could be a list of instances or could produce them lazily.
Because of this, the indexing was moved into `DataIterator` (see ["Indexing"](#indexing) below),
and the tensorization was moved into a new `Batch` abstraction (see ["Batch"](#batch) below).

## `Vocabulary`

As `Dataset` no longer exists, we replaced `Vocabulary.from_dataset()`
with `Vocabulary.from_instances()`, which accepts an `Iterable[Instance]`.
In particular, you'd most likely call this with the results of one or more calls
to `DatasetReader.read()`.

## `Batch`

To handle tensorization,
0.4.0 introduces the notion of a `Batch`,
which is basically just a list of `Instance`s.
In particular, a `Batch` knows how to compute padding
for its instances and convert them to tensors.

(Previously, we had `Dataset` doing double-duty
 to represent both full datasets and batches.)

## Indexing

In order to convert an `Instance` to tensors,
its fields must be _indexed_ by some vocabulary.

Previously, a `Dataset` contained all instances in memory,
and so you would call `Dataset.index_instances()`
after you loaded your data
and then have indexed instances ever after.

That doesn't work for lazy datasets, whose instances
are loaded into memory (and then discarded) as needed.

Accordingly, we moved the indexing functionality into
`DataIterator.index_with()`, which hangs onto the
`Vocabulary` you provide it and indexes each `Instance`
as it's iterated over.

Furthermore, each `Instance` now knows whether it's been indexed,
so in the eager case (when all instances stay in memory),
the indexing only happens in the first iteration.
This means that the first pass through your dataset will be slower
than subsequent ones.
