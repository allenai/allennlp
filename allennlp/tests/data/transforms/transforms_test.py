from allennlp.common.testing import AllenNlpTestCase

from allennlp.data import transforms
from allennlp.data import Vocabulary
from allennlp.data.fields import MetadataField
from torch.utils.data import IterableDataset

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

    def test_stop_after(self):

        stop_after = transforms.StopAfter(3)
        transformed = stop_after(self.instances)

        stopped = [instance for instance in transformed]

        assert stopped == [
            self.instances[0],
            self.instances[1],
            self.instances[2],
        ]

        stop_after = transforms.StopAfter(3)
        batch = transforms.Batch(2)
        transformed = batch(stop_after(self.instances))

        batches = [batch for batch in transformed]
        assert batches == [
            [self.instances[0], self.instances[1]],
            [self.instances[2]]
        ]
