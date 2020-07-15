import itertools
from typing import Iterable, Iterator, Optional, List, Any, Callable, Union
import logging
import os
from pathlib import Path
import warnings

from filelock import FileLock, Timeout
import jsonpickle
import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.common import Tqdm, util
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import CacheFile
from allennlp.common.registrable import Registrable

logger = logging.getLogger(__name__)


class AllennlpDataset(Dataset):
    """
    An `AllennlpDataset` is created by calling `.read()` on a non-lazy `DatasetReader`.
    It's essentially just a thin wrapper around a list of instances.
    """

    def __init__(self, instances: List[Instance], vocab: Vocabulary = None):
        self.instances = instances
        self.vocab = vocab

    def __getitem__(self, idx) -> Instance:
        if self.vocab is not None:
            self.instances[idx].index_fields(self.vocab)
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)

    def __iter__(self) -> Iterator[Instance]:
        """
        Even though it's not necessary to implement this because Python can infer
        this method from `__len__` and `__getitem__`, this helps with type-checking
        since `AllennlpDataset` can be considered an `Iterable[Instance]`.
        """
        yield from self.instances

    def index_with(self, vocab: Vocabulary):
        self.vocab = vocab


class AllennlpLazyDataset(IterableDataset):
    """
    An `AllennlpLazyDataset` is created by calling `.read()` on a lazy `DatasetReader`.

    # Parameters

    instance_generator : `Callable[[str], Iterable[Instance]]`
        A factory function that creates an iterable of `Instance`s from a file path.
        This is usually just `DatasetReader._instance_iterator`.
    file_path : `str`
        The path to pass to the `instance_generator` function.
    vocab : `Vocab`, optional (default = `None`)
        An optional vocab. This can also be set later with the `.index_with` method.
    """

    def __init__(
        self,
        instance_generator: Callable[[str], Iterable[Instance]],
        file_path: str,
        vocab: Vocabulary = None,
    ) -> None:
        super().__init__()
        self._instance_generator = instance_generator
        self._file_path = file_path
        self.vocab = vocab

    def __iter__(self) -> Iterator[Instance]:
        for instance in self._instance_generator(self._file_path):
            if self.vocab is not None:
                instance.index_fields(self.vocab)
            yield instance

    def index_with(self, vocab: Vocabulary):
        self.vocab = vocab


class DatasetReader(Registrable):
    """
    A `DatasetReader` knows how to turn a file containing a dataset into a collection
    of `Instances`.  To implement your own, just override the `_read(file_path)` method
    to return an `Iterable` of the instances. This could be a list containing the instances
    or a lazy generator that returns them one at a time.

    All parameters necessary to `_read` the data apart from the filepath should be passed
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

        !!! NOTE
            It is the _caller's_ responsibility to make sure that this directory is
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

    manual_multi_process_sharding : `bool`, optional (default=`False`)
        This is similar to the `manual_distributed_sharding` parameter, but applies to
        multi-process data loading. By default, when this reader is used by a multi-process
        data loader (i.e. a `DataLoader` with `num_workers > 1`), each worker will
        filter out all but a subset of the instances that are needed so that you
        don't end up with duplicates.

        !!! NOTE
            **There is really no benefit of using a multi-process
            `DataLoader` unless you can specifically implement a faster sharding mechanism
            within `_read()`**. In that case you should set `manual_multi_process_sharding`
            to `True`.

    """

    CACHE_FILE_LOCK_TIMEOUT: int = 10
    """
    The number of seconds to wait for the lock on a cache file to become available.
    """

    def __init__(
        self,
        lazy: bool = False,
        cache_directory: Optional[str] = None,
        max_instances: Optional[int] = None,
        manual_distributed_sharding: bool = False,
        manual_multi_process_sharding: bool = False,
    ) -> None:
        self.lazy = lazy
        self.max_instances = max_instances
        self._cache_directory: Optional[Path] = None
        if cache_directory:
            self._cache_directory = Path(cache_directory)
            os.makedirs(self._cache_directory, exist_ok=True)
        self.manual_distributed_sharding = manual_distributed_sharding
        self.manual_multi_process_sharding = manual_multi_process_sharding

    def read(self, file_path: Union[Path, str]) -> Union[AllennlpDataset, AllennlpLazyDataset]:
        """
        Returns an dataset containing all the instances that can be read from the file path.

        If `self.lazy` is `False`, this eagerly reads all instances from `self._read()`
        and returns an `AllennlpDataset`.

        If `self.lazy` is `True`, this returns an `AllennlpLazyDataset`, which internally
        relies on the generator created from `self._read()` to lazily produce `Instance`s.
        In this case your implementation of `_read()` must also be lazy
        (that is, not load all instances into memory at once), otherwise
        you will get a `ConfigurationError`.

        In either case, the returned `Iterable` can be iterated
        over multiple times. It's unlikely you want to override this function,
        but if you do your result should likewise be repeatedly iterable.
        """
        if not isinstance(file_path, str):
            file_path = str(file_path)

        lazy = getattr(self, "lazy", None)

        if lazy is None:
            warnings.warn(
                "DatasetReader.lazy is not set, "
                "did you forget to call the superclass constructor?",
                UserWarning,
            )

        if lazy:
            return AllennlpLazyDataset(self._instance_iterator, file_path)
        else:
            cache_file: Optional[str] = None
            if self._cache_directory:
                cache_file = self._get_cache_location_for_file_path(file_path)

            if cache_file is not None and os.path.exists(cache_file):
                try:
                    # Try to acquire a lock just to make sure another process isn't in the middle
                    # of writing to the cache.
                    cache_file_lock = FileLock(
                        cache_file + ".lock", timeout=self.CACHE_FILE_LOCK_TIMEOUT
                    )
                    cache_file_lock.acquire()
                    # We make an assumption here that if we can obtain the lock, no one will
                    # be trying to write to the file anymore, so it should be safe to release the lock
                    # before reading so that other processes can also read from it.
                    cache_file_lock.release()
                    logger.info("Reading instances from cache %s", cache_file)
                    instances = self._instances_from_cache_file(cache_file)
                except Timeout:
                    logger.warning(
                        "Failed to acquire lock on dataset cache file within %d seconds. "
                        "Cannot use cache to read instances.",
                        self.CACHE_FILE_LOCK_TIMEOUT,
                    )
                    instances = self._multi_worker_islice(self._read(file_path))
            else:
                instances = self._multi_worker_islice(self._read(file_path))

            # Then some validation.
            if not isinstance(instances, list):
                instances = list(instances)

            if not instances:
                raise ConfigurationError(
                    "No instances were read from the given filepath {}. "
                    "Is the path correct?".format(file_path)
                )

            # And finally we try writing to the cache.
            if cache_file is not None and not os.path.exists(cache_file):
                if self.max_instances is not None:
                    # But we don't write to the cache when max_instances is specified.
                    logger.warning(
                        "Skipping writing to data cache since max_instances was specified."
                    )
                elif util.is_distributed() or (get_worker_info() and get_worker_info().num_workers):
                    # We also shouldn't write to the cache if there's more than one process loading
                    # instances since each worker only receives a partial share of the instances.
                    logger.warning(
                        "Can't cache data instances when there are multiple processes loading data"
                    )
                else:
                    try:
                        with FileLock(cache_file + ".lock", timeout=self.CACHE_FILE_LOCK_TIMEOUT):
                            self._instances_to_cache_file(cache_file, instances)
                    except Timeout:
                        logger.warning(
                            "Failed to acquire lock on dataset cache file within %d seconds. "
                            "Cannot write to cache.",
                            self.CACHE_FILE_LOCK_TIMEOUT,
                        )

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
            yield from self._multi_worker_islice(cache_file, self.deserialize_instance)

    def _instances_to_cache_file(self, cache_filename, instances) -> None:
        # We serialize to a temp file first in case anything goes wrong while
        # writing to cache (e.g., the computer shuts down unexpectedly).
        # Then we just copy the file over to `cache_filename`.
        with CacheFile(cache_filename, mode="w+") as cache_handle:
            logger.info("Caching instances to temp file %s", cache_handle.name)
            for instance in Tqdm.tqdm(instances, desc="caching instances"):
                cache_handle.write(self.serialize_instance(instance) + "\n")

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
        return jsonpickle.loads(string.strip())  # type: ignore

    def _multi_worker_islice(
        self,
        iterable: Iterable[Any],
        transform: Optional[Callable[[Any], Instance]] = None,
        ensure_lazy: bool = False,
    ) -> Iterable[Instance]:
        """
        Helper method that determines which raw instances to skip based on the current
        node rank (for distributed training) and worker ID (for multi-process data loading).

        # Parameters

        iterable : `Iterable[Any]`
            An iterable that yields raw data that can be transformed into `Instance`s
            through the `transform` function.
        transform : `Optional[Callable[[Any], Instance]]`, optional (default = `None`)
            An optional function that will be applied to the raw data generated
            by `iterable` to create `Instance`s. This is used, e.g., when reading
            cached data.
        ensure_lazy : `bool`, optional (default = `False`)
            If `True`, a `ConfigurationError` error will be raised if `iterable`
            is a list instead of a lazy generator type.

        # Returns

        `Iterable[Instance]`
        """
        if ensure_lazy and isinstance(iterable, (list, tuple)):
            raise ConfigurationError("For a lazy dataset reader, _read() must return a generator")

        wrap_with_tqdm = True
        start_index = 0
        step_size = 1
        if not self.manual_distributed_sharding and util.is_distributed():
            start_index = dist.get_rank()
            step_size = dist.get_world_size()
        worker_info = None if self.manual_multi_process_sharding else get_worker_info()
        if worker_info:
            warnings.warn(
                "Using multi-process data loading without setting "
                "DatasetReader.manual_multi_process_sharding to True.\n"
                "Did you forget to set this?\n"
                "If you're not handling the multi-process sharding logic within your "
                "_read() method, there is probably no benefit to using more than one "
                "worker.",
                UserWarning,
            )
            # Scale `start_index` by `num_workers`, then shift by worker `id`.
            start_index *= worker_info.num_workers
            start_index += worker_info.id
            # Scale `step_size` by `num_workers`.
            step_size *= worker_info.num_workers
            if worker_info.id > 0:
                # We only want to log with tqdm from the main loader process.
                wrap_with_tqdm = False

        islice = itertools.islice(iterable, start_index, self.max_instances, step_size)
        if wrap_with_tqdm:
            islice = Tqdm.tqdm(islice, desc="reading instances")

        if transform is not None:
            return (transform(x) for x in islice)
        return islice

    def _instance_iterator(self, file_path: str) -> Iterable[Instance]:
        cache_file: Optional[str] = None
        if self._cache_directory:
            cache_file = self._get_cache_location_for_file_path(file_path)

        if cache_file is not None and os.path.exists(cache_file):
            cache_file_lock = FileLock(cache_file + ".lock", timeout=self.CACHE_FILE_LOCK_TIMEOUT)
            try:
                cache_file_lock.acquire()
                # We make an assumption here that if we can obtain the lock, no one will
                # be trying to write to the file anymore, so it should be safe to release the lock
                # before reading so that other processes can also read from it.
                cache_file_lock.release()
                logger.info("Reading instances from cache %s", cache_file)
                with open(cache_file) as data_file:
                    yield from self._multi_worker_islice(
                        data_file, transform=self.deserialize_instance
                    )
            except Timeout:
                logger.warning(
                    "Failed to acquire lock on dataset cache file within %d seconds. "
                    "Cannot use cache to read instances.",
                    self.CACHE_FILE_LOCK_TIMEOUT,
                )
                yield from self._multi_worker_islice(self._read(file_path), ensure_lazy=True)
        elif cache_file is not None and not os.path.exists(cache_file):
            instances = self._multi_worker_islice(self._read(file_path), ensure_lazy=True)
            # The cache file doesn't exist so we'll try writing to it.
            if self.max_instances is not None:
                # But we don't write to the cache when max_instances is specified.
                logger.warning("Skipping writing to data cache since max_instances was specified.")
                yield from instances
            elif util.is_distributed() or (get_worker_info() and get_worker_info().num_workers):
                # We also shouldn't write to the cache if there's more than one process loading
                # instances since each worker only receives a partial share of the instances.
                logger.warning(
                    "Can't cache data instances when there are multiple processes loading data"
                )
                yield from instances
            else:
                try:
                    with FileLock(cache_file + ".lock", timeout=self.CACHE_FILE_LOCK_TIMEOUT):
                        with CacheFile(cache_file, mode="w+") as cache_handle:
                            logger.info("Caching instances to temp file %s", cache_handle.name)
                            for instance in instances:
                                cache_handle.write(self.serialize_instance(instance) + "\n")
                                yield instance
                except Timeout:
                    logger.warning(
                        "Failed to acquire lock on dataset cache file within %d seconds. "
                        "Cannot write to cache.",
                        self.CACHE_FILE_LOCK_TIMEOUT,
                    )
                    yield from instances
        else:
            # No cache.
            yield from self._multi_worker_islice(self._read(file_path), ensure_lazy=True)
