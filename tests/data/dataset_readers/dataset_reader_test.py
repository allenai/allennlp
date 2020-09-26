from collections import deque
import os
import shutil
from typing import Optional, NamedTuple, List

from filelock import FileLock
import pytest
import torch.distributed as dist

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common import util as common_util
from allennlp.common.checks import ConfigurationError
from allennlp.data import Instance
from allennlp.data.dataloader import PyTorchDataLoader
from allennlp.data.dataset_readers import (
    dataset_reader,
    DatasetReader,
    TextClassificationJsonReader,
)
from allennlp.data.dataset_readers.dataset_reader import AllennlpLazyDataset
from allennlp.data.fields import LabelField


class TestDatasetReader(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.cache_directory = str(AllenNlpTestCase.FIXTURES_ROOT / "data_cache" / "with_prefix")

    def teardown_method(self):
        super().teardown_method()
        if os.path.exists(self.cache_directory):
            shutil.rmtree(self.cache_directory)

    def test_lazy_dataset_can_be_iterated_through_multiple_times(self):
        data_file = (
            AllenNlpTestCase.FIXTURES_ROOT
            / "data"
            / "text_classification_json"
            / "imdb_corpus.jsonl"
        )
        reader = TextClassificationJsonReader(lazy=True)
        instances = reader.read(data_file)
        assert isinstance(instances, AllennlpLazyDataset)

        first_pass_instances = list(instances)
        assert len(first_pass_instances) > 2
        second_pass_instances = list(instances)
        assert first_pass_instances == second_pass_instances

    def test_read_only_creates_cache_file_once(self):
        data_file = (
            AllenNlpTestCase.FIXTURES_ROOT
            / "data"
            / "text_classification_json"
            / "imdb_corpus.jsonl"
        )
        reader = TextClassificationJsonReader(cache_directory=self.cache_directory)
        cache_file = reader._get_cache_location_for_file_path(str(data_file))

        # The first read will create the cache.
        reader.read(data_file)
        assert os.path.exists(cache_file)
        with open(cache_file, "r") as in_file:
            cache_contents = in_file.read()
        # The second and all subsequent reads should _use_ the cache, not modify it.  I looked
        # into checking file modification times, but this test will probably be faster than the
        # granularity of `os.path.getmtime()` (which only returns values in seconds).
        reader.read(data_file)
        reader.read(data_file)
        reader.read(data_file)
        reader.read(data_file)
        with open(cache_file, "r") as in_file:
            final_cache_contents = in_file.read()
        assert cache_contents == final_cache_contents

    @pytest.mark.parametrize("lazy", (True, False))
    def test_caching_works_with_lazy_reading(self, caplog, lazy: bool):
        data_file = (
            AllenNlpTestCase.FIXTURES_ROOT
            / "data"
            / "text_classification_json"
            / "imdb_corpus.jsonl"
        )
        snli_copy_file = str(data_file) + ".copy"
        shutil.copyfile(data_file, snli_copy_file)
        reader = TextClassificationJsonReader(lazy=lazy, cache_directory=self.cache_directory)
        cache_file = reader._get_cache_location_for_file_path(snli_copy_file)

        # The call to read() will give us an _iterator_.  We'll iterate over it multiple times,
        # and the caching behavior should change as we go.
        assert not os.path.exists(cache_file)
        instances = reader.read(snli_copy_file)

        # The first iteration will create the cache
        first_pass_instances = []
        for instance in instances:
            first_pass_instances.append(instance)
        assert "Caching instances to temp file" in " ".join([rec.message for rec in caplog.records])
        assert os.path.exists(cache_file)

        # Now we _remove_ the data file, to be sure we're reading from the cache.
        os.remove(snli_copy_file)
        caplog.clear()
        instances = reader.read(snli_copy_file)
        second_pass_instances = []
        for instance in instances:
            second_pass_instances.append(instance)
        assert "Reading instances from cache" in " ".join([rec.message for rec in caplog.records])

        # We should get the same instances both times.
        assert len(first_pass_instances) == len(second_pass_instances)
        for instance, cached_instance in zip(first_pass_instances, second_pass_instances):
            assert instance.fields == cached_instance.fields

        # And just to be super paranoid, in case the second pass somehow bypassed the cache
        # because of a bug that's hard to detect, we'll read the
        # instances from the cache with a non-lazy iterator and make sure they're the same.
        reader = TextClassificationJsonReader(lazy=False, cache_directory=self.cache_directory)
        cached_instances = reader.read(snli_copy_file)
        assert len(first_pass_instances) == len(cached_instances)
        for instance, cached_instance in zip(first_pass_instances, cached_instances):
            assert instance.fields == cached_instance.fields

    @pytest.mark.parametrize("lazy", (True, False))
    def test_caching_skipped_when_lock_not_acquired(self, caplog, lazy: bool):
        data_file = (
            AllenNlpTestCase.FIXTURES_ROOT
            / "data"
            / "text_classification_json"
            / "imdb_corpus.jsonl"
        )
        reader = TextClassificationJsonReader(lazy=lazy, cache_directory=self.cache_directory)
        reader.CACHE_FILE_LOCK_TIMEOUT = 1
        cache_file = reader._get_cache_location_for_file_path(str(data_file))

        with FileLock(cache_file + ".lock"):
            # Right now we hold the lock on the cache, so the reader shouldn't
            # be able to write to it. It will wait for 1 second (because that's what
            # we set the timeout to be), and then just read the instances as normal.
            caplog.clear()
            instances = list(reader.read(data_file))
            assert "Failed to acquire lock" in caplog.text
            assert instances

        # We didn't write to the cache because we couldn't acquire the file lock.
        assert not os.path.exists(cache_file)

        # Now we'll write to the cache and then try the same thing again, this
        # time making sure that we can still successfully read without the cache
        # when the lock can't be acquired.
        deque(reader.read(data_file), maxlen=1)
        assert os.path.exists(cache_file)

        with FileLock(cache_file + ".lock"):
            # Right now we hold the lock on the cache, so the reader shouldn't
            # be able to write to it. It will wait for 1 second (because that's what
            # we set the timeout to be), and then just read the instances as normal.
            caplog.clear()
            instances = list(reader.read(data_file))
            assert "Failed to acquire lock" in caplog.text
            assert instances

    @pytest.mark.parametrize("lazy", (True, False))
    def test_caching_skipped_with_distributed_training(self, caplog, monkeypatch, lazy):
        monkeypatch.setattr(common_util, "is_distributed", lambda: True)
        monkeypatch.setattr(dist, "get_rank", lambda: 0)
        monkeypatch.setattr(dist, "get_world_size", lambda: 1)

        data_file = (
            AllenNlpTestCase.FIXTURES_ROOT
            / "data"
            / "text_classification_json"
            / "imdb_corpus.jsonl"
        )
        reader = TextClassificationJsonReader(lazy=lazy, cache_directory=self.cache_directory)
        cache_file = reader._get_cache_location_for_file_path(str(data_file))

        deque(reader.read(data_file), maxlen=1)
        assert not os.path.exists(cache_file)
        assert "Can't cache data instances when there are multiple processes" in caplog.text

    def test_caching_with_lazy_reader_in_multi_process_loader(self):
        data_file = (
            AllenNlpTestCase.FIXTURES_ROOT
            / "data"
            / "text_classification_json"
            / "imdb_corpus.jsonl"
        )
        reader = TextClassificationJsonReader(lazy=True, cache_directory=self.cache_directory)
        deque(
            PyTorchDataLoader(reader.read(data_file), collate_fn=lambda b: b[0], num_workers=2),
            maxlen=0,
        )

        # We shouldn't write to the cache when the data is being loaded from multiple
        # processes.
        cache_file = reader._get_cache_location_for_file_path(str(data_file))
        assert not os.path.exists(cache_file)

        # But try again from the main process and we should see the cache file.
        instances = list(reader.read(data_file))
        assert instances
        assert os.path.exists(cache_file)

        # Reading again from a multi-process loader should read from the cache.
        new_instances = list(
            PyTorchDataLoader(reader.read(data_file), collate_fn=lambda b: b[0], num_workers=2)
        )
        assert len(instances) == len(new_instances)

    @pytest.mark.parametrize("lazy", (True, False))
    def test_max_instances(self, lazy):
        data_file = (
            AllenNlpTestCase.FIXTURES_ROOT
            / "data"
            / "text_classification_json"
            / "imdb_corpus.jsonl"
        )
        reader = TextClassificationJsonReader(max_instances=2, lazy=lazy)
        instances = reader.read(data_file)
        instance_count = sum(1 for _ in instances)
        assert instance_count == 2

    @pytest.mark.parametrize("num_workers", (0, 1, 2))
    def test_max_instances_with_multi_process_loader(self, num_workers):
        data_file = (
            AllenNlpTestCase.FIXTURES_ROOT
            / "data"
            / "text_classification_json"
            / "imdb_corpus.jsonl"
        )
        reader = TextClassificationJsonReader(max_instances=2, lazy=True)
        instances = list(
            PyTorchDataLoader(
                reader.read(data_file), collate_fn=lambda b: b[0], num_workers=num_workers
            )
        )
        assert len(instances) == 2

    @pytest.mark.parametrize("lazy", (True, False))
    def test_cached_max_instances(self, lazy):
        data_file = (
            AllenNlpTestCase.FIXTURES_ROOT
            / "data"
            / "text_classification_json"
            / "imdb_corpus.jsonl"
        )

        # If we try reading with max instances, it shouldn't write to the cache.
        reader = TextClassificationJsonReader(
            cache_directory=self.cache_directory, lazy=lazy, max_instances=2
        )
        instances = list(reader.read(data_file))
        assert len(instances) == 2

        cache_file = reader._get_cache_location_for_file_path(str(data_file))
        assert not os.path.exists(cache_file)

        # Now reading again with no max_instances specified should create the cache.
        reader = TextClassificationJsonReader(cache_directory=self.cache_directory, lazy=lazy)
        instances = list(reader.read(data_file))
        assert len(instances) > 2
        assert os.path.exists(cache_file)

        # The second read should only return two instances, even though it's from the cache.
        reader = TextClassificationJsonReader(
            cache_directory=self.cache_directory, max_instances=2, lazy=lazy
        )
        instances = list(reader.read(data_file))
        assert len(instances) == 2


class MockWorkerInfo(NamedTuple):
    id: int
    num_workers: int


class MockDatasetReader(DatasetReader):
    def _read(self, file_path):
        for i in range(10):
            yield self.text_to_instance(i)

    def text_to_instance(self, index: int):  # type: ignore
        return Instance({"index": LabelField(index, skip_indexing=True)})


@pytest.mark.parametrize(
    "node_rank, world_size, worker_id, num_workers, max_instances, expected_result",
    [
        (None, None, None, None, None, list(range(10))),
        (None, None, None, None, 5, list(range(5))),
        (None, None, None, None, 12, list(range(10))),
        (None, None, 0, 1, None, list(range(10))),
        (None, None, 0, 2, None, [0, 2, 4, 6, 8]),
        (None, None, 1, 2, None, [1, 3, 5, 7, 9]),
        (None, None, 0, 2, 5, [0, 2, 4]),
        (None, None, 1, 2, 5, [1, 3]),
        (0, 1, None, None, None, list(range(10))),
        (0, 2, None, None, None, [0, 2, 4, 6, 8]),
        (1, 2, None, None, None, [1, 3, 5, 7, 9]),
        (0, 2, None, None, 5, [0, 2, 4]),
        (1, 2, None, None, 5, [1, 3]),
        (0, 2, 0, 2, None, [0, 4, 8]),
        (0, 2, 1, 2, None, [1, 5, 9]),
        (1, 2, 0, 2, None, [2, 6]),
        (1, 2, 1, 2, None, [3, 7]),
        (0, 2, 0, 2, 5, [0, 4]),
    ],
)
def test_instance_slicing(
    monkeypatch,
    node_rank: Optional[int],
    world_size: Optional[int],
    worker_id: Optional[int],
    num_workers: Optional[int],
    max_instances: Optional[int],
    expected_result: List[int],
):
    if node_rank is not None and world_size is not None:
        monkeypatch.setattr(common_util, "is_distributed", lambda: True)
        monkeypatch.setattr(dist, "get_rank", lambda: node_rank)
        monkeypatch.setattr(dist, "get_world_size", lambda: world_size)

    if worker_id is not None and num_workers is not None:
        monkeypatch.setattr(
            dataset_reader, "get_worker_info", lambda: MockWorkerInfo(worker_id, num_workers)
        )

    reader = MockDatasetReader(max_instances=max_instances)
    result = list((x["index"].label for x in reader.read("the-path-doesnt-matter")))  # type: ignore

    assert result == expected_result


class BadLazyReader(DatasetReader):
    def _read(self, file_path):
        return [self.text_to_instance(i) for i in range(10)]

    def text_to_instance(self, index: int):  # type: ignore
        return Instance({"index": LabelField(index, skip_indexing=True)})


def test_config_error_when_lazy_reader_returns_list():
    reader = BadLazyReader(lazy=True)
    with pytest.raises(ConfigurationError, match="must return a generator"):
        deque(reader.read("path"), maxlen=0)


class BadReaderReadsNothing(DatasetReader):
    def _read(self, file_path):
        return []

    def text_to_instance(self, index: int):  # type: ignore
        return Instance({"index": LabelField(index, skip_indexing=True)})


def test_config_error_when_reader_returns_no_instances():
    reader = BadReaderReadsNothing()
    with pytest.raises(ConfigurationError, match="No instances were read"):
        deque(reader.read("path"), maxlen=0)


class BadReaderForgetsToSetLazy(DatasetReader):
    def __init__(self):
        pass

    def _read(self, file_path):
        for i in range(10):
            yield self.text_to_instance(i)

    def text_to_instance(self, index: int):  # type: ignore
        return Instance({"index": LabelField(index, skip_indexing=True)})


def warning_when_reader_has_no_lazy_set():
    with pytest.warns(UserWarning, match="DatasetReader.lazy is not set"):
        reader = BadReaderForgetsToSetLazy()
        reader.read("path")
