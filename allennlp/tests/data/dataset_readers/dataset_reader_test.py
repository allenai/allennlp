# pylint: disable=invalid-name,no-self-use,protected-access
import os
import shutil

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import SnliReader
from allennlp.data.dataset_readers.dataset_reader import _CachedLazyInstances


class DatasetReaderTest(AllenNlpTestCase):
    def test_read_creates_cache_file_when_not_present(self):
        snli_file = self.FIXTURES_ROOT / "data" / "snli.jsonl"
        suffix = ".cache"
        cache_file = str(snli_file) + suffix
        reader = SnliReader(cache_suffix=suffix)

        try:
            assert not os.path.exists(cache_file)
            reader.read(snli_file)
            assert os.path.exists(cache_file)
        finally:
            os.remove(cache_file)

    def test_read_uses_existing_cache_file_when_present(self):
        snli_file = self.FIXTURES_ROOT / "data" / "snli.jsonl"
        snli_copy_file = str(snli_file) + ".copy"
        shutil.copyfile(snli_file, snli_copy_file)
        suffix = ".cache"
        cache_file = snli_copy_file + suffix
        reader = SnliReader(cache_suffix=suffix)

        # The first read will create the cache.
        instances = reader.read(snli_copy_file)
        # Now we _remove_ the data file, to be sure we're reading from the cache.
        os.remove(snli_copy_file)
        cached_instances = reader.read(snli_copy_file)
        try:
            # We should get the same instances both times.
            assert len(instances) == len(cached_instances)
            for instance, cached_instance in zip(instances, cached_instances):
                assert instance.fields == cached_instance.fields
        finally:
            os.remove(cache_file)

    def test_read_only_creates_cache_file_once(self):
        snli_file = self.FIXTURES_ROOT / "data" / "snli.jsonl"
        suffix = ".cache"
        cache_file = str(snli_file) + suffix
        reader = SnliReader(cache_suffix=suffix)

        try:
            # The first read will create the cache.
            reader.read(snli_file)
            assert os.path.exists(cache_file)
            with open(cache_file, 'r') as in_file:
                cache_contents = in_file.read()
            # The second and all subsequent reads should _use_ the cache, not modify it.  I looked
            # into checking file modification times, but this test will probably be faster than the
            # granularity of `os.path.getmtime()` (which only returns values in seconds).
            reader.read(snli_file)
            reader.read(snli_file)
            reader.read(snli_file)
            reader.read(snli_file)
            with open(cache_file, 'r') as in_file:
                final_cache_contents = in_file.read()
            assert cache_contents == final_cache_contents
        finally:
            os.remove(cache_file)

    def test_caching_works_with_lazy_reading(self):
        snli_file = self.FIXTURES_ROOT / "data" / "snli.jsonl"
        snli_copy_file = str(snli_file) + ".copy"
        shutil.copyfile(snli_file, snli_copy_file)
        suffix = ".cache"
        cache_file = snli_copy_file + suffix
        reader = SnliReader(lazy=True, cache_suffix=suffix)

        try:
            # The call to read() will give us an _iterator_.  We'll iterate over it multiple times,
            # and the caching behavior should change as we go.
            instances = reader.read(snli_copy_file)
            assert isinstance(instances, _CachedLazyInstances)

            # The first iteration will create the cache
            assert not os.path.exists(cache_file)
            first_pass_instances = []
            for instance in instances:
                first_pass_instances.append(instance)
            assert os.path.exists(cache_file)

            # Now we _remove_ the data file, to be sure we're reading from the cache.
            os.remove(snli_copy_file)
            second_pass_instances = []
            for instance in instances:
                second_pass_instances.append(instance)

            # We should get the same instances both times.
            assert len(first_pass_instances) == len(second_pass_instances)
            for instance, cached_instance in zip(first_pass_instances, second_pass_instances):
                assert instance.fields == cached_instance.fields

            # And just to be super paranoid, in case the second pass somehow bypassed the cache
            # because of a bug in `_CachedLazyInstance` that's hard to detect, we'll read the
            # instances from the cache with a non-lazy iterator and make sure they're the same.
            reader = SnliReader(lazy=False, cache_suffix=suffix)
            cached_instances = reader.read(snli_copy_file)
            assert len(first_pass_instances) == len(cached_instances)
            for instance, cached_instance in zip(first_pass_instances, cached_instances):
                assert instance.fields == cached_instance.fields
        finally:
            if os.path.exists(cache_file):
                os.remove(cache_file)
            if os.path.exists(snli_copy_file):
                os.remove(snli_copy_file)
