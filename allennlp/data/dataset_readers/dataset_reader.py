from typing import Iterable, Iterator, Callable
import logging
import os
import pathlib

import jsonpickle

from allennlp.data.instance import Instance
from allennlp.common import Tqdm, util
from allennlp.common.checks import ConfigurationError
from allennlp.common.registrable import Registrable

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class _LazyInstances(Iterable):
    """
    An ``Iterable`` that just wraps a thunk for generating instances and calls it for
    each call to ``__iter__``.
    """
    def __init__(self,
                 instance_generator: Callable[[], Iterable[Instance]],
                 cache_file: str = None,
                 deserialize: Callable[[str], Instance] = None,
                 serialize: Callable[[Instance], str] = None) -> None:
        super().__init__()
        self.instance_generator = instance_generator
        self.cache_file = cache_file
        self.deserialize = deserialize
        self.serialize = serialize

    def __iter__(self) -> Iterator[Instance]:
        # Case 1: Use cached instances
        if self.cache_file is not None and os.path.exists(self.cache_file):
            with open(self.cache_file) as data_file:
                for line in data_file:
                    yield self.deserialize(line)
        # Case 2: Need to cache instances
        elif self.cache_file is not None:
            with open(self.cache_file, 'w') as data_file:
                for instance in self.instance_generator():
                    data_file.write(self.serialize(instance))
                    data_file.write("\n")
                    yield instance
        # Case 3: No cache
        else:
            instances = self.instance_generator()
            if isinstance(instances, list):
                raise ConfigurationError("For a lazy dataset reader, _read() must return a generator")
            yield from instances


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
    """
    def __init__(self, lazy: bool = False) -> None:
        self.lazy = lazy
        self._cache_directory: pathlib.Path = None

    def cache_data(self, cache_directory: str) -> None:
        """
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
        """
        self._cache_directory = pathlib.Path(cache_directory)
        os.makedirs(self._cache_directory, exist_ok=True)

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

        if lazy is None:
            logger.warning("DatasetReader.lazy is not set, "
                           "did you forget to call the superclass constructor?")

        if self._cache_directory:
            cache_file = self._get_cache_location_for_file_path(file_path)
        else:
            cache_file = None

        if lazy:
            return _LazyInstances(lambda: self._read(file_path),
                                  cache_file,
                                  self.deserialize_instance,
                                  self.serialize_instance)
        else:
            # First we read the instances, either from a cache or from the original file.
            if cache_file and os.path.exists(cache_file):
                instances = self._instances_from_cache_file(cache_file)
            else:
                instances = self._read(file_path)

            # Then some validation.
            if not isinstance(instances, list):
                instances = [instance for instance in Tqdm.tqdm(instances)]
            if not instances:
                raise ConfigurationError("No instances were read from the given filepath {}. "
                                         "Is the path correct?".format(file_path))

            # And finally we write to the cache if we need to.
            if cache_file and not os.path.exists(cache_file):
                logger.info(f"Caching instances to {cache_file}")
                self._instances_to_cache_file(cache_file, instances)

            return instances

    def _get_cache_location_for_file_path(self, file_path: str) -> str:
        return str(self._cache_directory / util.flatten_filename(str(file_path)))

    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        Reads the instances from the given file_path and returns them as an
        `Iterable` (which could be a list or could be a generator).
        You are strongly encouraged to use a generator, so that users can
        read a dataset in a lazy way, if they so choose.
        """
        raise NotImplementedError

    def _instances_from_cache_file(self, cache_filename: str) -> Iterable[Instance]:
        with open(cache_filename, 'r') as cache_file:
            for line in cache_file:
                yield self.deserialize_instance(line.strip())

    def _instances_to_cache_file(self, cache_filename, instances) -> None:
        with open(cache_filename, 'w') as cache:
            for instance in Tqdm.tqdm(instances):
                cache.write(self.serialize_instance(instance) + '\n')

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
        # pylint: disable=no-self-use
        return jsonpickle.dumps(instance)

    def deserialize_instance(self, string: str) -> Instance:
        """
        Deserializes an ``Instance`` from a string.  We use this when reading processed data from a
        cache.

        The default implementation is to use ``jsonpickle``.  If you would like some other format
        for your pre-processed data, override this method.
        """
        # pylint: disable=no-self-use
        return jsonpickle.loads(string)  # type: ignore
