# allennlp.data.dataset_readers.dataset_reader

## DatasetReader
```python
DatasetReader(self, lazy:bool=False) -> None
```

A ``DatasetReader`` knows how to turn a file containing a dataset into a collection
of ``Instance`` s.  To implement your own, just override the `_read(file_path)` method
to return an ``Iterable`` of the instances. This could be a list containing the instances
or a lazy generator that returns them one at a time.

All parameters necessary to _read the data apart from the filepath should be passed
to the constructor of the ``DatasetReader``.

Parameters
----------
lazy : ``bool``, optional (default=False)
    If this is true, ``instances()`` will return an object whose ``__iter__`` method
    reloads the dataset each time it's called. Otherwise, ``instances()`` returns a list.

### cache_data
```python
DatasetReader.cache_data(self, cache_directory:str) -> None
```

When you call this method, we will use this directory to store a cache of already-processed
``Instances`` in every file passed to :func:`read`, serialized as one string-formatted
``Instance`` per line.  If the cache file for a given ``file_path`` exists, we read the
``Instances`` from the cache instead of re-processing the data (using
:func:`deserialize_instance`).  If the cache file does `not` exist, we will `create` it on
our first pass through the data (using :func:`serialize_instance`).

IMPORTANT CAVEAT: It is the `caller's` responsibility to make sure that this directory is
unique for any combination of code and parameters that you use.  That is, if you call this
method, we will use any existing cache files in that directory `regardless of the
parameters you set for this DatasetReader!`  If you use our commands, the ``Train`` command
is responsible for calling this method and ensuring that unique parameters correspond to
unique cache directories.  If you don't use our commands, that is your responsibility.

### read
```python
DatasetReader.read(self, file_path:str) -> Iterable[allennlp.data.instance.Instance]
```

Returns an ``Iterable`` containing all the instances
in the specified dataset.

If ``self.lazy`` is False, this calls ``self._read()``,
ensures that the result is a list, then returns the resulting list.

If ``self.lazy`` is True, this returns an object whose
``__iter__`` method calls ``self._read()`` each iteration.
In this case your implementation of ``_read()`` must also be lazy
(that is, not load all instances into memory at once), otherwise
you will get a ``ConfigurationError``.

In either case, the returned ``Iterable`` can be iterated
over multiple times. It's unlikely you want to override this function,
but if you do your result should likewise be repeatedly iterable.

### text_to_instance
```python
DatasetReader.text_to_instance(self, *inputs) -> allennlp.data.instance.Instance
```

Does whatever tokenization or processing is necessary to go from textual input to an
``Instance``.  The primary intended use for this is with a
:class:`~allennlp.predictors.predictor.Predictor`, which gets text input as a JSON
object and needs to process it to be input to a model.

The intent here is to share code between :func:`_read` and what happens at
model serving time, or any other time you want to make a prediction from new data.  We need
to process the data in the same way it was done at training time.  Allowing the
``DatasetReader`` to process new text lets us accomplish this, as we can just call
``DatasetReader.text_to_instance`` when serving predictions.

The input type here is rather vaguely specified, unfortunately.  The ``Predictor`` will
have to make some assumptions about the kind of ``DatasetReader`` that it's using, in order
to pass it the right information.

### serialize_instance
```python
DatasetReader.serialize_instance(self, instance:allennlp.data.instance.Instance) -> str
```

Serializes an ``Instance`` to a string.  We use this for caching the processed data.

The default implementation is to use ``jsonpickle``.  If you would like some other format
for your pre-processed data, override this method.

### deserialize_instance
```python
DatasetReader.deserialize_instance(self, string:str) -> allennlp.data.instance.Instance
```

Deserializes an ``Instance`` from a string.  We use this when reading processed data from a
cache.

The default implementation is to use ``jsonpickle``.  If you would like some other format
for your pre-processed data, override this method.

