import unittest
from typing import List

from _pytest.monkeypatch import MonkeyPatch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError

from allennlp.data.dataset import Batch
from allennlp.data.iterators.bucket_iterator import BucketIterator, BucketIteratorShim
from allennlp.data.iterators.transform_iterator import TransformIterator
from allennlp.tests.data.iterators.basic_iterator_test import IteratorTest


def _collocate_patch(self, batch: List) -> Batch:

    # If we've added a Batch() into the pipeline,
    # this is a length one list containing a batch.
    # So we unpack it.
    if len(batch) == 1:
        batch = list(batch[0])
    allennlp_batch = Batch(batch)

    # We might have already done this - but it doesn't matter if we have,
    # because if so it's a no-op.
    allennlp_batch.index_instances(self.vocab)
    return allennlp_batch


class TestBucketIteratorStub(IteratorTest):
    def setUp(self):
        super().setUp()
        self.monkeypatch = MonkeyPatch()

        self.monkeypatch.setattr(TransformIterator, "_collocate", _collocate_patch)

    def tearDown(self):
        self.monkeypatch.undo()
        super().tearDown()

    def test_create_batches_groups_correctly(self):
        iterator = BucketIteratorShim(
            batch_size=2, padding_noise=0, sorting_keys=[("text", "num_tokens")]
        )
        iterator.index_with(self.vocab)
        batches = list(iterator._create_batches(self.instances, shuffle=False))
        grouped_instances = [batch.instances for batch in batches]
        assert grouped_instances == [
            [self.instances[4], self.instances[2]],
            [self.instances[0], self.instances[1]],
            [self.instances[3]],
        ]

    def test_create_batches_groups_correctly_with_max_instances(self):
        # If we knew all the instances, the correct order is 4 -> 2 -> 0 -> 1 -> 3.
        # Here max_instances_in_memory is 3, so we load instances [0, 1, 2]
        # and then bucket them by size into batches of size 2 to get [2, 0] -> [1].
        # Then we load the remaining instances and bucket them by size to get [4, 3].
        iterator = BucketIteratorShim(
            batch_size=2,
            padding_noise=0,
            sorting_keys=[("text", "num_tokens")],
            max_instances_in_memory=3,
        )
        iterator.index_with(self.vocab)
        for test_instances in (self.instances, self.lazy_instances):
            batches = list(iterator._create_batches(test_instances, shuffle=False))
            grouped_instances = [batch.instances for batch in batches]

            assert grouped_instances == [
                [self.instances[2], self.instances[0]],
                [self.instances[1]],
                [self.instances[4], self.instances[3]],
            ]

    def test_biggest_batch_first_works(self):
        iterator = BucketIteratorShim(
            batch_size=2,
            padding_noise=0,
            sorting_keys=[("text", "num_tokens")],
            biggest_batch_first=True,
        )
        iterator.index_with(self.vocab)
        batches = list(iterator._create_batches(self.instances, shuffle=False))
        grouped_instances = [batch.instances for batch in batches]
        assert grouped_instances == [
            [self.instances[3]],
            [self.instances[0], self.instances[1]],
            [self.instances[4], self.instances[2]],
        ]

    def test_from_params(self):

        params = Params({})
        # Construction with no sorting keys is allowed.
        iterator = BucketIteratorShim.from_params(params)

        sorting_keys = [("s1", "nt"), ("s2", "nt2")]
        params["sorting_keys"] = sorting_keys
        iterator = BucketIteratorShim.from_params(params)

        assert iterator._batch_size == 32

        params = Params(
            {
                "sorting_keys": sorting_keys,
                "padding_noise": 0.5,
                "biggest_batch_first": True,
                "batch_size": 100,
                "skip_smaller_batches": True,
            }
        )
        iterator = BucketIterator.from_params(params)
        assert iterator._batch_size == 100

    def test_bucket_iterator_maximum_samples_per_batch(self):
        iterator = BucketIteratorShim(
            batch_size=3,
            padding_noise=0,
            sorting_keys=[("text", "num_tokens")],
            maximum_samples_per_batch=["num_tokens", 9],
        )
        iterator.index_with(self.vocab)
        batches = list(iterator._create_batches(self.instances, shuffle=False))
        stats = self.get_batches_stats(batches)

        # ensure all instances are in a batch
        assert stats["total_instances"] == len(self.instances)

        # ensure correct batch sizes
        assert stats["batch_lengths"] == [2, 2, 1]

        # ensure correct sample sizes (<= 9)
        assert stats["sample_sizes"] == [6, 8, 9]

    def test_maximum_samples_per_batch_packs_tightly(self):
        token_counts = [10, 4, 3]
        test_instances = self.create_instances_from_token_counts(token_counts)

        iterator = BucketIteratorShim(
            batch_size=3,
            padding_noise=0,
            sorting_keys=[("text", "num_tokens")],
            maximum_samples_per_batch=["num_tokens", 11],
        )
        iterator.index_with(self.vocab)
        batches = list(iterator._create_batches(test_instances, shuffle=False))
        stats = self.get_batches_stats(batches)

        # ensure all instances are in a batch
        assert stats["total_instances"] == len(test_instances)

        # ensure correct batch sizes
        assert stats["batch_lengths"] == [2, 1]

        # ensure correct sample sizes (<= 11)
        assert stats["sample_sizes"] == [8, 10]

    def test_skip_smaller_batches_works(self):
        iterator = BucketIteratorShim(
            batch_size=2,
            padding_noise=0,
            sorting_keys=[("text", "num_tokens")],
            skip_smaller_batches=True,
        )
        iterator.index_with(self.vocab)
        batches = list(iterator._create_batches(self.instances, shuffle=False))
        stats = self.get_batches_stats(batches)

        # all batches have length batch_size
        assert all(batch_len == 2 for batch_len in stats["batch_lengths"])

        # we should have lost one instance by skipping the last batch
        assert stats["total_instances"] == len(self.instances) - 1
