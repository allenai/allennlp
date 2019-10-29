import pytest

from allennlp.common.testing import AllenNlpTestCase

from allennlp.data import transforms
from allennlp.data import Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data.fields import MetadataField
from torch.utils.data import IterableDataset, DataLoader

from allennlp.tests.data.iterators.basic_iterator_test import IteratorTest


class TransformsTest(IteratorTest):
    def test_batch_transform(self):

        transformed = transforms.Batch(2)(self.instances)

        batches = [batch for batch in transformed]

        assert batches == [
            [self.instances[0], self.instances[1]],
            [self.instances[2], self.instances[3]],
            [self.instances[4]],
        ]

        # batch should work with nested lists.
        transformed = transforms.Batch(2)([self.instances, self.instances])
        batches = [batch for batch in transformed]
        assert batches == [
            [self.instances[0], self.instances[1]],
            [self.instances[2], self.instances[3]],
            [self.instances[4]],
            [self.instances[0], self.instances[1]],
            [self.instances[2], self.instances[3]],
            [self.instances[4]],
        ]

    def test_max_instances_in_memory(self):
        # This is the same as the batching test, because these
        # classes do the same thing.
        transformed = transforms.MaxInstancesInMemory(2)(self.instances)

        batches = [batch for batch in transformed]

        assert batches == [
            [self.instances[0], self.instances[1]],
            [self.instances[2], self.instances[3]],
            [self.instances[4]],
        ]

        # max_instances_in_memory should work with nested lists.
        transformed = transforms.MaxInstancesInMemory(2)([self.instances, self.instances])
        batches = [batch for batch in transformed]
        assert batches == [
            [self.instances[0], self.instances[1]],
            [self.instances[2], self.instances[3]],
            [self.instances[4]],
            [self.instances[0], self.instances[1]],
            [self.instances[2], self.instances[3]],
            [self.instances[4]],
        ]

    def test_index_indexes_instances(self):

        vocab = Vocabulary.from_instances(self.instances)
        index = transforms.Index(vocab)
        transformed = index(self.instances)

        for instance in transformed:
            assert instance.indexed

        # Now check that when Index is called on batched
        # input, it doesn't flatten the input back out.
        # This essentially tests that transform_batch is working
        # correctly.
        batch = transforms.Batch(2)
        transformed = index(batch(self.instances))
        batches = [[instance for instance in batch] for batch in transformed]

        assert batches == [
            [self.instances[0], self.instances[1]],
            [self.instances[2], self.instances[3]],
            [self.instances[4]],
        ]

        for batch in batches:
            for instance in batch:
                assert instance.indexed

    def test_epoch_tracker_adds_metadata(self):

        track = transforms.EpochTracker()
        transformed = track(self.instances)

        for instance in transformed:
            assert instance.fields["epoch_num"].metadata == 0

        # Second epoch should have a new epoch_number
        for instance in transformed:
            assert instance.fields["epoch_num"].metadata == 1

    def test_skip_smaller_than(self):

        batch = transforms.Batch(2)
        skip = transforms.SkipSmallerThan(2)

        transformed = skip(batch(self.instances))
        batches = [batch for batch in transformed]

        # We should have skipped the left over amount
        assert batches == [
            [self.instances[0], self.instances[1]],
            [self.instances[2], self.instances[3]],
        ]
        # SkipSmallerThan cannot be called on unbached input:
        with pytest.raises(NotImplementedError):
            [x for x in skip(self.instances)]

    def test_stop_after(self):
        stop_after = transforms.StopAfter(3)
        transformed = stop_after(self.instances)

        stopped = [instance for instance in transformed]

        assert stopped == [self.instances[0], self.instances[1], self.instances[2]]

        stop_after = transforms.StopAfter(3)
        batch = transforms.Batch(2)
        transformed = batch(stop_after(self.instances))

        batches = [batch for batch in transformed]
        assert batches == [[self.instances[0], self.instances[1]], [self.instances[2]]]

        # StopAfter should have the same behaviour for batches
        # as it does for instances.
        stop_after = transforms.StopAfter(2)
        batch = transforms.Batch(2)
        transformed = stop_after(batch(self.instances))

        batches = [batch for batch in transformed]
        assert batches == [
            [self.instances[0], self.instances[1]],
            [self.instances[2], self.instances[3]],
        ]

    def test_sort_by_padding(self):

        index = transforms.Index(self.vocab)
        batch = transforms.Batch(2)
        sort = transforms.SortByPadding(sorting_keys=[("text", "num_tokens")], padding_noise=0)

        pipeline = transforms.Compose([index, sort, batch])
        transformed = pipeline(self.instances)
        batches = [batch for batch in transformed]

        assert batches == [
            [self.instances[4], self.instances[2]],
            [self.instances[0], self.instances[1]],
            [self.instances[3]],
        ]

    def test_maximum_samples_per_batch(self):

        index = transforms.Index(self.vocab)
        batch = transforms.Batch(3)
        sort = transforms.SortByPadding(sorting_keys=[("text", "num_tokens")], padding_noise=0)
        max_samples = transforms.MaxSamplesPerBatch(("num_tokens", 9))

        pipeline = transforms.Compose([index, sort, max_samples, batch])

        transformed = pipeline(self.instances)
        # Make batches compatible with the test helper by
        # wrapping them in an allennlp batch.
        batches = [Batch(batch) for batch in transformed]

        stats = self.get_batches_stats(batches)
        # ensure all instances are in a batch
        assert stats["total_instances"] == len(self.instances)
        # ensure correct batch sizes
        assert stats["batch_lengths"] == [2, 2, 1]
        # ensure correct sample sizes (<= 9)
        assert stats["sample_sizes"] == [6, 8, 9]

    def test_biggest_batch_first(self):

        index = transforms.Index(self.vocab)
        batch = transforms.Batch(2)
        sort = transforms.SortByPadding(sorting_keys=[("text", "num_tokens")], padding_noise=0)
        biggest = transforms.BiggestBatchFirst()

        pipeline = transforms.Compose([index, sort, batch, biggest])
        batches = [batch for batch in pipeline(self.instances)]

        assert batches == [
            [self.instances[3]],
            [self.instances[0], self.instances[1]],
            [self.instances[4], self.instances[2]],
        ]

    def test_fork(self):
        batch = transforms.Batch(2)
        fork = transforms.Fork()

        # If we fork after we batch, we should get batches in order.
        pipeline = transforms.Compose([batch, fork])
        loader = DataLoader(
            pipeline(self.instances), num_workers=2, collate_fn=lambda batch: batch[0]
        )

        batches = {tuple(self.instances.index(instance) for instance in batch) for batch in loader}
        assert batches == {(0, 1), (2, 3), (4,)}

        # If we fork before we batch, we should get batches where the instances are modulo the
        # number of workers
        pipeline = transforms.Compose([fork, batch])
        loader = DataLoader(
            pipeline(self.instances), num_workers=2, collate_fn=lambda batch: batch[0]
        )

        batches = {tuple(self.instances.index(instance) for instance in batch) for batch in loader}
        assert batches == {(0, 2), (1, 3), (4,)}
