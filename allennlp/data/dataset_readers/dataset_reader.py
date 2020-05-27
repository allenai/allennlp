import itertools
from typing import Iterable, Iterator, Callable, Optional, List
import logging
import os
import pathlib

import jsonpickle
from torch.utils.data import Dataset, IterableDataset

from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.common import Tqdm, util
from allennlp.common.checks import ConfigurationError
from allennlp.common.registrable import Registrable

logger = logging.getLogger(__name__)


class AllennlpDataset(Dataset):
    def __init__(self, instances: List[Instance], vocab: Vocabulary = None):
        self.instances = instances
        self.vocab = vocab

    def __getitem__(self, idx):
        if self.vocab is not None:
            self.instances[idx].index_fields(self.vocab)
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)

    def index_with(self, vocab: Vocabulary):
        self.vocab = vocab


class AllennlpLazyDataset(IterableDataset):
    def __init__(
        self,
        deserialize: Callable[[str], Instance] = None,
        serialize: Callable[[Instance], str] = None,
        vocab: Vocabulary = None,
    ) -> None:
        super().__init__()
        self.deserialize = deserialize
        self.serialize = serialize
        self.vocab = vocab

    def index_with(self, vocab: Vocabulary):
        self.vocab = vocab

    def __len__(self):
        """
        We rely in a couple of places that calling len on the dataloader
        (which in turn calls len on the dataset) doesn't raise an error.
        In the case that you have an IterableDataset and you call len, the pytorch dataloader
        actually spits out a warning - but we need actually calling it to not crash.
        """
        return 1


class _LazyInstances(AllennlpLazyDataset):
    """
    An `Iterable` that just wraps a thunk for generating instances and calls it for
    each call to `__iter__`.
    """

    def __init__(
        self,
        instance_generator: Callable[[str], Iterable[Instance]],
        file_path: str,
        cache_file: str = None,
        deserialize: Callable[[str], Instance] = None,
        serialize: Callable[[Instance], str] = None,
        vocab: Vocabulary = None,
    ) -> None:
        super().__init__(deserialize, serialize, vocab)
        self.instance_generator = instance_generator
        self.file_path = file_path
        self.cache_file = cache_file

    def __iter__(self) -> Iterator[Instance]:
        # Case 1: Use cached instances
        if self.cache_file is not None and os.path.exists(self.cache_file):
            with open(self.cache_file) as data_file:
                for line in data_file:
                    instance = self.deserialize(line)
                    if self.vocab is not None:
                        instance.index_fields(self.vocab)
                    yield instance

        # Case 2: Need to cache instances
        elif self.cache_file is not None:
            with open(self.cache_file, "w") as data_file:
                for instance in self.instance_generator(self.file_path):
                    data_file.write(self.serialize(instance))
                    data_file.write("\n")
                    if self.vocab is not None:
                        instance.index_fields(self.vocab)
                    yield instance
        # Case 3: No cache
        else:
            instances = self.instance_generator(self.file_path)
            if isinstance(instances, list):
                raise ConfigurationError(
                    "For a lazy dataset reader, _read() must return a generator"
                )
            for instance in instances:
                if self.vocab is not None:
                    instance.index_fields(self.vocab)
                yield instance


class _MaxLazyInstances(AllennlpLazyDataset):
    def __init__(self, inner: AllennlpLazyDataset, max_instances: int) -> None:
        super().__init__()
        self.inner = inner
        self.max_instances = max_instances

    def __iter__(self) -> Iterator[Instance]:
        return itertools.islice(iter(self.inner), self.max_instances)

    def index_with(self, vocab: Vocabulary):
        self.inner.index_with(vocab)

    def __len__(self):
        return len(self.inner)


class _DistributedLazyInstances(AllennlpLazyDataset):
    def __init__(self, inner: AllennlpLazyDataset) -> None:
        super().__init__()
        self.inner = inner

    def __iter__(self) -> Iterator[Instance]:
        from torch import distributed

        logger.info(
            "Returning instances i%%%d==%d", distributed.get_world_size(), distributed.get_rank()
        )
        return itertools.islice(
            iter(self.inner), distributed.get_rank(), None, distributed.get_world_size()
        )

    def index_with(self, vocab: Vocabulary):
        self.inner.index_with(vocab)

    def __len__(self):
        return len(self.inner)


class DatasetReader(Registrable):
    """
    A `DatasetReader` knows how to turn a file containing a dataset into a collection
    of `Instances`.  To implement your own, just override the `_read(file_path)` method
    to return an `Iterable` of the instances. This could be a list containing the instances
    or a lazy generator that returns them one at a time.

    All parameters necessary to _read the data apart from the filepath should be passed
    to the constructor of the `DatasetReader`.

    # Parameters

    lazy : `bool`, optional (default=`False`)
        If this is true, `instances()` will return an object whose `__iter__` method
        reloads the dataset each time it's called. Otherwise, `instances()` returns a list.
    cache_directory : `str`, optional (default=`None`)
        If given, we will use this directory to store a cache of already-processed `Instances` in
        every file passed to :func:`read`, serialized (by default, though you can override this) as
        one string-formatted `Instance` per line.  If the cache file for a given `file_path` exists,
        we read the `Instances` from the cache instead of re-processing the data (using
        :func:`_instances_from_cache_file`).  If the cache file does _not_ exist, we will _create_
        it on our first pass through the data (using :func:`_instances_to_cache_file`).

        IMPORTANT CAVEAT: It is the _caller's_ responsibility to make sure that this directory is
        unique for any combination of code and parameters that you use.  That is, if you pass a
        directory here, we will use any existing cache files in that directory _regardless of the
        parameters you set for this DatasetReader!_
    max_instances : `int`, optional (default=`None`)
        If given, will stop reading after this many instances. This is a useful setting for debugging.
        Setting this disables caching.
    manual_distributed_sharding: `bool`, optional (default=`False`)
        By default, when used in a distributed setting, `DatasetReader` makes sure that each
        worker process only receives a subset of the data. It does this by reading the whole
        dataset in each worker, but filtering out the instances that are not needed. If you
        can implement a faster mechanism that only reads part of the data, set this to True,
        and do the sharding yourself.
    """

    def __init__(
        self,
        lazy: bool = False,
        cache_directory: Optional[str] = None,
        max_instances: Optional[int] = None,
        manual_distributed_sharding: bool = False,
    ) -> None:
        self.lazy = lazy
        self.max_instances = max_instances
        if cache_directory:
            self._cache_directory = pathlib.Path(cache_directory)
            os.makedirs(self._cache_directory, exist_ok=True)
        else:
            self._cache_directory = None
        self.manual_distributed_sharding = manual_distributed_sharding

    def read(self, file_path: str) -> Dataset:
        """
        Returns an `Iterable` containing all the instances
        in the specified dataset.

        If `self.lazy` is False, this calls `self._read()`,
        ensures that the result is a list, then returns the resulting list.

        If `self.lazy` is True, this returns an object whose
        `__iter__` method calls `self._read()` each iteration.
        In this case your implementation of `_read()` must also be lazy
        (that is, not load all instances into memory at once), otherwise
        you will get a `ConfigurationError`.

        In either case, the returned `Iterable` can be iterated
        over multiple times. It's unlikely you want to override this function,
        but if you do your result should likewise be repeatedly iterable.
        """
        lazy = getattr(self, "lazy", None)

        if lazy is None:
            logger.warning(
                "DatasetReader.lazy is not set, "
                "did you forget to call the superclass constructor?"
            )

        if self._cache_directory:
            cache_file = self._get_cache_location_for_file_path(file_path)
        else:
            cache_file = None

        if lazy:
            lazy_instances: AllennlpLazyDataset = _LazyInstances(
                self._read,
                file_path,
                cache_file,
                self.deserialize_instance,
                self.serialize_instance,
            )
            if self.max_instances is not None:
                lazy_instances = _MaxLazyInstances(lazy_instances, self.max_instances)
            if not self.manual_distributed_sharding and util.is_distributed():
                lazy_instances = _DistributedLazyInstances(lazy_instances)
            return lazy_instances
        else:
            # First we read the instances, either from a cache or from the original file.
            if cache_file and os.path.exists(cache_file):
                instances = self._instances_from_cache_file(cache_file)
            else:
                instances = self._read(file_path)

            if self.max_instances is not None:
                if isinstance(instances, list):
                    instances = instances[: self.max_instances]
                else:
                    instances = itertools.islice(instances, 0, self.max_instances)
            if not self.manual_distributed_sharding and util.is_distributed():
                from torch import distributed

                logger.info(
                    "Returning instances i%%%d==%d",
                    distributed.get_world_size(),
                    distributed.get_rank(),
                )
                instances = itertools.islice(
                    instances, distributed.get_rank(), None, distributed.get_world_size()
                )

            # Then some validation.
            if not isinstance(instances, list):
                instances = [instance for instance in Tqdm.tqdm(instances)]
            if not instances:
                raise ConfigurationError(
                    "No instances were read from the given filepath {}. "
                    "Is the path correct?".format(file_path)
                )

            # And finally we write to the cache if we need to.
            if (
                self.max_instances is None
                and not util.is_distributed()
                and cache_file is not None
                and not os.path.exists(cache_file)
            ):
                logger.info(f"Caching instances to {cache_file}")
                self._instances_to_cache_file(cache_file, instances)

            return AllennlpDataset(instances)

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
        with open(cache_filename, "r") as cache_file:
            for line in cache_file:
                yield self.deserialize_instance(line.strip())

    def _instances_to_cache_file(self, cache_filename, instances) -> None:
        with open(cache_filename, "w") as cache:
            for instance in Tqdm.tqdm(instances):
                cache.write(self.serialize_instance(instance) + "\n")

    def text_to_instance(self, *inputs) -> Instance:
        """
        Does whatever tokenization or processing is necessary to go from textual input to an
        `Instance`.  The primary intended use for this is with a
        :class:`~allennlp.predictors.predictor.Predictor`, which gets text input as a JSON
        object and needs to process it to be input to a model.

        The intent here is to share code between :func:`_read` and what happens at
        model serving time, or any other time you want to make a prediction from new data.  We need
        to process the data in the same way it was done at training time.  Allowing the
        `DatasetReader` to process new text lets us accomplish this, as we can just call
        `DatasetReader.text_to_instance` when serving predictions.

        The input type here is rather vaguely specified, unfortunately.  The `Predictor` will
        have to make some assumptions about the kind of `DatasetReader` that it's using, in order
        to pass it the right information.
        """
        raise NotImplementedError

    def serialize_instance(self, instance: Instance) -> str:
        """
        Serializes an `Instance` to a string.  We use this for caching the processed data.

        The default implementation is to use `jsonpickle`.  If you would like some other format
        for your pre-processed data, override this method.
        """

        return jsonpickle.dumps(instance)

    def deserialize_instance(self, string: str) -> Instance:
        """
        Deserializes an `Instance` from a string.  We use this when reading processed data from a
        cache.

        The default implementation is to use `jsonpickle`.  If you would like some other format
        for your pre-processed data, override this method.
        """

        return jsonpickle.loads(string)  # type: ignore
