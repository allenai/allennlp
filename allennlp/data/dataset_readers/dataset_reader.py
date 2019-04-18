from typing import Iterable, Iterator, Callable
import logging
import os

import jsonpickle

from allennlp.data.instance import Instance
from allennlp.common import Tqdm
from allennlp.common.checks import ConfigurationError
from allennlp.common.registrable import Registrable

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class _LazyInstances(Iterable):
    """
    An ``Iterable`` that just wraps a thunk for generating instances and calls it for
    each call to ``__iter__``.
    """
    def __init__(self, instance_generator: Callable[[], Iterator[Instance]]) -> None:
        super().__init__()
        self.instance_generator = instance_generator

    def __iter__(self) -> Iterator[Instance]:
        instances = self.instance_generator()
        if isinstance(instances, list):
            raise ConfigurationError("For a lazy dataset reader, _read() must return a generator")
        return instances


class _CachedLazyInstances(Iterable):
    """
    Like ``_LazyInstances``, but we take the actual input file that the instances come from, as
    well as a cache file.  If the cache file exists, we read from that instead of from the main
    file.  If the cache file `doesn't` exist, we create it on our first pass, and read it during
    subsequent passes.
    """
    def __init__(self,
                 read_from_file_method: Callable[[str], Iterator[Instance]],
                 read_from_cache_method: Callable[[str], Iterator[Instance]],
                 serialize_instance_method: Callable[[Instance], str],
                 file_path: str,
                 cache_file: str) -> None:
        super().__init__()
        self.read_from_file_method = read_from_file_method
        self.read_from_cache_method = read_from_cache_method
        self.serialize_instance_method = serialize_instance_method
        self.file_path = file_path
        self.cache_file = cache_file

    def __iter__(self) -> Iterator[Instance]:
        if os.path.exists(self.cache_file):
            return self.read_from_cache_method(self.cache_file)
        else:
            def iterator():
                with open(self.cache_file, 'w') as cache_file:
                    for instance in self.read_from_file_method(self.file_path):
                        cache_file.write(self.serialize_instance_method(instance) + '\n')
                        yield instance
            return iterator()


class DatasetReader(Registrable):
    """
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
    cache_suffix : ``str``, optional (default=None)
        When provided, we add this suffix to each path passed to :func:`read` to get a location of
        a file containing a cache of already-processed ``Instances`` in that path, serialized as
        one string-formatted ``Instance`` per line.  If this file exists, we read the ``Instances``
        from the cache instead of re-processing the data (using :func:`deserialize_instance`).  If
        this file does `not` exist, we will `create` it on our first pass through the data (using
        :func:`serialize_instance`).

        IMPORTANT CAVEAT: if you change parameters in your dataset reader that are supposed to
        affect how instances are processed, you `must` change this suffix, or clear the cache!
        Otherwise you will think you are using your new parameters, but you are actually using
        whatever parameters created the cache.
    """
    def __init__(self, lazy: bool = False, cache_suffix: str = None) -> None:
        self.lazy = lazy
        self.cache_suffix = cache_suffix

    def read(self, file_path: str) -> Iterable[Instance]:
        """
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
        """
        lazy = getattr(self, 'lazy', None)
        cache_suffix = getattr(self, 'cache_suffix', None)
        if cache_suffix:
            cache_file = str(file_path) + cache_suffix
            return self._read_cached(file_path, cache_file, lazy)

        if lazy is None:
            logger.warning("DatasetReader.lazy is not set, "
                           "did you forget to call the superclass constructor?")

        if lazy:
            return _LazyInstances(lambda: iter(self._read(file_path)))
        else:
            instances = self._read(file_path)
            if not isinstance(instances, list):
                instances = [instance for instance in Tqdm.tqdm(instances)]
            if not instances:
                raise ConfigurationError("No instances were read from the given filepath {}. "
                                         "Is the path correct?".format(file_path))
            return instances

    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        Reads the instances from the given file_path and returns them as an
        `Iterable` (which could be a list or could be a generator).
        You are strongly encouraged to use a generator, so that users can
        read a dataset in a lazy way, if they so choose.
        """
        raise NotImplementedError

    def _read_cached(self, file_path: str, cache_file: str, lazy: bool) -> Iterable[Instance]:
        if lazy:
            return _CachedLazyInstances(self._read,
                                        self._instances_from_cache_file,
                                        self.serialize_instance,
                                        file_path,
                                        cache_file)
        if os.path.exists(cache_file):
            # We already have cached instances; just read them.
            logger.info(f"Reading cached instances from {cache_file}")
            instances = self._instances_from_cache_file(cache_file)
            if not isinstance(instances, list):
                instances = [instance for instance in Tqdm.tqdm(instances)]
            if not instances:
                raise ConfigurationError("No instances were read from the given filepath {}. "
                                         "Is the path correct?".format(cache_file))
            return instances
        else:
            # We don't have any cached instances yet, so we create them before returning.
            instances = self._read(file_path)
            if not isinstance(instances, list):
                instances = [instance for instance in Tqdm.tqdm(instances)]
            if not instances:
                raise ConfigurationError("No instances were read from the given filepath {}. "
                                         "Is the path correct?".format(file_path))
            logger.info(f"Caching instances to {cache_file}")
            with open(cache_file, 'w') as cache_file:
                for instance in Tqdm.tqdm(instances):
                    cache_file.write(self.serialize_instance(instance) + '\n')
            return instances

    def _instances_from_cache_file(self, cache_filename: str) -> Iterable[Instance]:
        with open(cache_filename, 'r') as cache_file:
            for line in cache_file:
                yield self.deserialize_instance(line.strip())

    def text_to_instance(self, *inputs) -> Instance:
        """
        Does whatever tokenization or processing is necessary to go from textual input to an
        ``Instance``.  The primary intended use for this is with a
        :class:`~allennlp.service.predictors.predictor.Predictor`, which gets text input as a JSON
        object and needs to process it to be input to a model.

        The intent here is to share code between :func:`_read` and what happens at
        model serving time, or any other time you want to make a prediction from new data.  We need
        to process the data in the same way it was done at training time.  Allowing the
        ``DatasetReader`` to process new text lets us accomplish this, as we can just call
        ``DatasetReader.text_to_instance`` when serving predictions.

        The input type here is rather vaguely specified, unfortunately.  The ``Predictor`` will
        have to make some assumptions about the kind of ``DatasetReader`` that it's using, in order
        to pass it the right information.
        """
        raise NotImplementedError

    def serialize_instance(self, instance: Instance) -> str:
        """
        Serializes an ``Instance`` to a string.  We use this for caching the processed data.

        The default implementation is to use ``jsonpickle``.  If you would like some other format
        for your pre-processed data, override this method.
        """
        return jsonpickle.dumps(instance)

    def deserialize_instance(self, string: str) -> Instance:
        """
        Deserializes an ``Instance`` from a string.  We use this when reading processed data from a
        cache.

        The default implementation is to use ``jsonpickle``.  If you would like some other format
        for your pre-processed data, override this method.
        """
        return jsonpickle.loads(string)  # type: ignore
