from typing import List, Iterable, Dict, Union
from collections import Counter

from _pytest.monkeypatch import MonkeyPatch

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Instance, Token, Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data.dataset_readers.dataset_reader import _LazyInstances
from allennlp.data.fields import TextField

from allennlp.data.iterators.basic_iterator import BasicIterator, BasicIteratorStub
from allennlp.data.iterators.transform_iterator import TransformIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer

from allennlp.tests.data.iterators.basic_iterator_test import IteratorTest


class LazyIterable:
    def __init__(self, instances):
        self._instances = instances

    def __iter__(self):
        return (instance for instance in self._instances)


class TestBasicIteratorStub(IteratorTest):
    def test_get_num_batches(self):
        # Lazy and instances per epoch not specified.
        assert BasicIteratorStub(batch_size=2).get_num_batches(self.lazy_instances) == 1
        # Lazy and instances per epoch specified.
        assert (
            BasicIteratorStub(batch_size=2, instances_per_epoch=21).get_num_batches(
                self.lazy_instances
            )
            == 11
        )
        # Not lazy and instances per epoch specified.
        assert (
            BasicIteratorStub(batch_size=2, instances_per_epoch=21).get_num_batches(self.instances)
            == 11
        )
        # Not lazy and instances per epoch not specified.
        assert BasicIteratorStub(batch_size=2).get_num_batches(self.instances) == 3

    # The BasicIterator should work the same for lazy and non lazy datasets,
    # so each remaining test runs over both.
    def test_yield_one_epoch_iterates_over_the_data_once(self):
        for test_instances in (self.instances, self.lazy_instances):
            iterator = BasicIteratorStub(batch_size=2)
            iterator.index_with(self.vocab)
            batches = list(iterator(test_instances, num_epochs=1))
            # We just want to get the single-token array for the text field in the instance.
            instances = [
                tuple(instance.detach().cpu().numpy())
                for batch in batches
                for instance in batch["text"]["tokens"]
            ]
            assert len(instances) == 5
            self.assert_instances_are_correct(instances)

    def test_call_iterates_over_data_forever(self):
        for test_instances in (self.instances, self.lazy_instances):
            iterator = BasicIteratorStub(batch_size=2)
            iterator.index_with(self.vocab)
            generator = iterator(test_instances)
            batches = [next(generator) for _ in range(18)]  # going over the data 6 times
            # We just want to get the single-token array for the text field in the instance.
            instances = [
                tuple(instance.detach().cpu().numpy())
                for batch in batches
                for instance in batch["text"]["tokens"]
            ]
            assert len(instances) == 5 * 6
            self.assert_instances_are_correct(instances)

    def test_epoch_tracking_when_one_epoch_at_a_time(self):
        iterator = BasicIteratorStub(batch_size=2, track_epoch=True)
        iterator.index_with(self.vocab)
        for epoch in range(10):
            for batch in iterator(self.instances, num_epochs=1):
                assert all(epoch_num == epoch for epoch_num in batch["epoch_num"])

    def test_epoch_tracking_forever(self):
        iterator = BasicIteratorStub(batch_size=2, track_epoch=True)
        iterator.index_with(self.vocab)

        it = iterator(self.instances, num_epochs=None)

        all_batches = [next(it) for _ in range(30)]

        assert len(all_batches) == 30
        for i, batch in enumerate(all_batches):
            # Should have 3 batches per epoch
            epoch = i // 3
            assert all(epoch_num == epoch for epoch_num in batch["epoch_num"])

    def test_epoch_tracking_multiple_epochs(self):
        iterator = BasicIteratorStub(batch_size=2, track_epoch=True)
        iterator.index_with(self.vocab)

        all_batches = list(iterator(self.instances, num_epochs=10))
        assert len(all_batches) == 10 * 3
        for i, batch in enumerate(all_batches):
            # Should have 3 batches per epoch
            epoch = i // 3
            assert all(epoch_num == epoch for epoch_num in batch["epoch_num"])


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


class TestBasicIteratorStubPatchRequired(IteratorTest):
    def setUp(self):
        super().setUp()
        self.monkeypatch = MonkeyPatch()

        self.monkeypatch.setattr(TransformIterator, "_collocate", _collocate_patch)

    def tearDown(self):
        self.monkeypatch.undo()
        super().tearDown()

    def test_create_batches_groups_correctly(self):

        for test_instances in (self.instances, self.lazy_instances):
            iterator = BasicIteratorStub(batch_size=2)
            iterator.index_with(self.vocab)
            batches = list(iterator._create_batches(test_instances, shuffle=False))
            grouped_instances = [batch.instances for batch in batches]
            assert grouped_instances == [
                [self.instances[0], self.instances[1]],
                [self.instances[2], self.instances[3]],
                [self.instances[4]],
            ]

    def test_few_instances_per_epoch(self):

        for test_instances in (self.instances, self.lazy_instances):
            iterator = BasicIteratorStub(batch_size=2, instances_per_epoch=3)
            iterator.index_with(self.vocab)
            # First epoch: 3 instances -> [2, 1]
            batches = list(iterator._create_batches(test_instances, shuffle=False))
            grouped_instances = [batch.instances for batch in batches]

            assert grouped_instances == [
                [self.instances[0], self.instances[1]],
                [self.instances[2]],
            ]
            # Second epoch: 3 instances -> [2, 1]
            batches = list(iterator._create_batches(test_instances, shuffle=False))
            grouped_instances = [batch.instances for batch in batches]
            assert grouped_instances == [
                [self.instances[3], self.instances[4]],
                [self.instances[0]],
            ]

            # Third epoch: 3 instances -> [2, 1]
            batches = list(iterator._create_batches(test_instances, shuffle=False))
            grouped_instances = [batch.instances for batch in batches]
            assert grouped_instances == [
                [self.instances[1], self.instances[2]],
                [self.instances[3]],
            ]

    def test_shuffle(self):
        for test_instances in (self.instances, self.lazy_instances):

            iterator = BasicIteratorStub(batch_size=2, instances_per_epoch=100)
            iterator.index_with(self.vocab)

            in_order_batches = list(iterator._create_batches(test_instances, shuffle=False))
            shuffled_batches = list(iterator._create_batches(test_instances, shuffle=True))

            assert len(in_order_batches) == len(shuffled_batches)

            # With 100 instances, shuffling better change the order.
            assert in_order_batches != shuffled_batches

            # But not the counts of the instances.
            in_order_counts = Counter(
                id(instance) for batch in in_order_batches for instance in batch
            )
            shuffled_counts = Counter(
                id(instance) for batch in shuffled_batches for instance in batch
            )
            assert in_order_counts == shuffled_counts

    def test_max_instances_in_memory(self):
        for test_instances in (self.instances, self.lazy_instances):
            iterator = BasicIteratorStub(batch_size=2, max_instances_in_memory=3)
            iterator.index_with(self.vocab)
            # One epoch: 5 instances -> [2, 1, 2]
            batches = list(iterator._create_batches(test_instances, shuffle=False))
            grouped_instances = [batch.instances for batch in batches]
            assert grouped_instances == [
                [self.instances[0], self.instances[1]],
                [self.instances[2]],
                [self.instances[3], self.instances[4]],
            ]

    def test_multiple_cursors(self):

        lazy_instances1 = _LazyInstances(lambda: (i for i in self.instances))
        lazy_instances2 = _LazyInstances(lambda: (i for i in self.instances))

        eager_instances1 = self.instances[:]
        eager_instances2 = self.instances[:]

        for instances1, instances2 in [
            (eager_instances1, eager_instances2),
            (lazy_instances1, lazy_instances2),
        ]:
            iterator = BasicIteratorStub(batch_size=1, instances_per_epoch=2)
            assert iterator._instances_per_epoch == 2
            iterator.index_with(self.vocab)

            # First epoch through dataset1
            batches = list(iterator._create_batches(instances1, shuffle=False))
            grouped_instances = [batch.instances for batch in batches]
            assert grouped_instances == [[self.instances[0]], [self.instances[1]]]

            # First epoch through dataset2
            batches = list(iterator._create_batches(instances2, shuffle=False))
            grouped_instances = [batch.instances for batch in batches]
            assert grouped_instances == [[self.instances[0]], [self.instances[1]]]

            # Second epoch through dataset1
            batches = list(iterator._create_batches(instances1, shuffle=False))
            grouped_instances = [batch.instances for batch in batches]
            assert grouped_instances == [[self.instances[2]], [self.instances[3]]]

            # Second epoch through dataset2
            batches = list(iterator._create_batches(instances2, shuffle=False))
            grouped_instances = [batch.instances for batch in batches]
            assert grouped_instances == [[self.instances[2]], [self.instances[3]]]

    def test_from_params(self):

        params = Params({})
        iterator = BasicIteratorStub.from_params(params)
        assert iterator._batch_size == 32  # default value

        params = Params({"batch_size": 10})
        iterator = BasicIteratorStub.from_params(params)
        assert iterator._batch_size == 10

    def test_maximum_samples_per_batch(self):
        for test_instances in (self.instances, self.lazy_instances):

            iterator = BasicIteratorStub(batch_size=3, maximum_samples_per_batch=["num_tokens", 9])
            iterator.index_with(self.vocab)
            batches = list(iterator._create_batches(test_instances, shuffle=False))
            stats = self.get_batches_stats(batches)

            # ensure all instances are in a batch
            assert stats["total_instances"] == len(self.instances)

            # ensure correct batch sizes
            assert stats["batch_lengths"] == [2, 1, 1, 1]

            # ensure correct sample sizes (<= 9)
            assert stats["sample_sizes"] == [8, 3, 9, 1]

    def test_maximum_samples_per_batch_packs_tightly(self):

        token_counts = [10, 4, 3]
        test_instances = self.create_instances_from_token_counts(token_counts)

        iterator = BasicIteratorStub(batch_size=3, maximum_samples_per_batch=["num_tokens", 11])
        iterator.index_with(self.vocab)
        batches = list(iterator._create_batches(test_instances, shuffle=False))
        stats = self.get_batches_stats(batches)

        # ensure all instances are in a batch
        assert stats["total_instances"] == len(token_counts)

        # ensure correct batch sizes
        assert stats["batch_lengths"] == [1, 2]

        # ensure correct sample sizes (<= 11)
        assert stats["sample_sizes"] == [10, 8]
