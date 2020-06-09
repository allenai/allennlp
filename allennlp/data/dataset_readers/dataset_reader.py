import itertools
from typing import Iterable, Iterator, Optional, List, Dict, Any, Union, Callable
import logging
import os
import pathlib

from filelock import FileLock, Timeout
import jsonpickle
import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.common import util
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
        instance_iterator_factory: Callable[[str], Iterable[Instance]],
        file_path: str,
        vocab: Vocabulary = None,
    ) -> None:
        super().__init__()
        self._instance_iterator_factory = instance_iterator_factory
        self._file_path = file_path
        self.vocab = vocab

    def __iter__(self) -> Iterator[Instance]:
        for instance in self._instance_iterator_factory(self._file_path):
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
    ) -> None:
        self.lazy = lazy
        self.max_instances = max_instances
        self._cache_directory: Optional[pathlib.Path] = None
        if cache_directory:
            self._cache_directory = pathlib.Path(cache_directory)
            os.makedirs(self._cache_directory, exist_ok=True)
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

        if lazy:
            return AllennlpLazyDataset(self.instance_iterator, file_path)

        instances = list(self.instance_iterator(file_path))
        if not instances:
            raise ConfigurationError(
                "No instances were read from the given filepath {}. "
                "Is the path correct?".format(file_path)
            )

        return AllennlpDataset(instances)

    def instance_iterator(self, file_path: str) -> Iterable[Instance]:
        cache_file: Optional[str] = None
        if self._cache_directory:
            cache_file = self._check_for_cache(file_path)

        if cache_file is not None:
            # Try to acquire lock on cache file.
            cache_file_lock = FileLock(cache_file + ".lock", timeout=self.CACHE_FILE_LOCK_TIMEOUT)
            try:
                cache_file_lock.acquire()
            except Timeout:
                logger.error(
                    "Failed to acquire lock on dataset cache file within {}s",
                    self.CACHE_FILE_LOCK_TIMEOUT,
                )
                raise

            if os.path.exists(cache_file):
                # If the file already exists and we've acquired the lock then it must
                # be up-to-date. Therefore we can release the lock and start reading from it.
                cache_file_lock.release()
                with open(cache_file) as data_file:
                    for line in itertools.islice(data_file, self.max_instances):
                        instance = self.deserialize_instance(line)
                        yield instance
            else:
                try:
                    with open(cache_file, "w") as data_file:
                        for instance in self._auto_islice(self._read(file_path)):
                            data_file.write(self.serialize_instance(instance))
                            data_file.write("\n")
                            yield instance
                except:  # noqa: E722
                    # If anything went wrong, the cache file will be corrupted, so we should
                    # remove it.
                    logger.warning("Removing dataset cache file '%s' due to exception", cache_file)
                    os.remove(cache_file)
                    raise
                finally:
                    # Release the lock no matter what.
                    cache_file_lock.release()
        else:
            # No cache, just start reading right away.
            for instance in self._auto_islice(self._read(file_path)):
                yield instance

    def _check_for_cache(self, file_path: str) -> str:
        node_rank: int = 0
        worker_id: int = 0
        world_size: int = 1
        num_workers: int = 1
        if util.is_distributed():
            node_rank = dist.get_rank()
            world_size = dist.get_world_size()
        worker_info = get_worker_info()
        if worker_info:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        # The cache file has to be unique for each data loader worker and each training node.
        node_worker_id_suffix = (
            f"_node_{node_rank}_of_{world_size}_worker_{worker_id}_of_{num_workers}"
        )
        cache_file_name = util.flatten_filename(str(file_path)) + node_worker_id_suffix
        return str(self._cache_directory / cache_file_name)

    def _read(self, file_path: str) -> Iterable[Union[Instance, Dict[str, Any]]]:
        """
        Reads the instances (or the data that can be turned into any instance) from the
        given file_path and returns them as an `Iterable`.

        The objects in the iterable should ideally just be dictionaries that hold the
        arguments to `self.text_to_instance`, but for backwards compatability with older
        versions of `AllenNLP`, the objects can also be `Instance`s. This will usually be
        less efficient though.

        You are strongly encouraged to use a generator as opposed to a "greedy" iterable
        type like a list.
        """
        raise NotImplementedError

    def text_to_instance(self, *args, **kwargs) -> Instance:
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

    def _auto_islice(
        self, iterable: Iterable[Union[Instance, Dict[str, Any]]]
    ) -> Iterable[Union[Instance, Dict[str, Any]]]:
        """
        Helper method that determines which raw instances to skip based on the current
        node rank (for distributed training) and worker ID (for multi-process data loading).
        """
        start_index = 0
        step_size = 1
        if not self.manual_distributed_sharding and util.is_distributed():
            start_index = dist.get_rank()
            step_size = dist.get_world_size()
        worker_info = get_worker_info()
        if worker_info:
            start_index += step_size * worker_info.id
            step_size *= worker_info.num_workers

        def ensure_instance(raw_instance: Union[Instance, Dict[str, Any]]):
            if isinstance(raw_instance, Instance):
                return raw_instance
            return self.text_to_instance(**raw_instance)

        return map(
            ensure_instance, itertools.islice(iterable, start_index, self.max_instances, step_size)
        )
