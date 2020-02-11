from typing import List, Iterable, Dict, Union
from collections import Counter

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Instance, Token, Vocabulary
from allennlp.data.batch import Batch
from allennlp.data.dataset_readers.dataset_reader import _LazyInstances
from allennlp.data.fields import TextField
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer


class LazyIterable:
    def __init__(self, instances):
        self._instances = instances

    def __iter__(self):
        return (instance for instance in self._instances)


class IteratorTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.token_indexers = {"tokens": SingleIdTokenIndexer()}
        self.vocab = Vocabulary()
        self.this_index = self.vocab.add_token_to_namespace("this")
        self.is_index = self.vocab.add_token_to_namespace("is")
        self.a_index = self.vocab.add_token_to_namespace("a")
        self.sentence_index = self.vocab.add_token_to_namespace("sentence")
        self.another_index = self.vocab.add_token_to_namespace("another")
        self.yet_index = self.vocab.add_token_to_namespace("yet")
        self.very_index = self.vocab.add_token_to_namespace("very")
        self.long_index = self.vocab.add_token_to_namespace("long")
        instances = [
            self.create_instance(["this", "is", "a", "sentence"]),
            self.create_instance(["this", "is", "another", "sentence"]),
            self.create_instance(["yet", "another", "sentence"]),
            self.create_instance(
                ["this", "is", "a", "very", "very", "very", "very", "long", "sentence"]
            ),
            self.create_instance(["sentence"]),
        ]

        self.instances = instances
        self.lazy_instances = LazyIterable(instances)

    def create_instance(self, str_tokens: List[str]):
        tokens = [Token(t) for t in str_tokens]
        instance = Instance({"text": TextField(tokens, self.token_indexers)})
        return instance

    def create_instances_from_token_counts(self, token_counts: List[int]) -> List[Instance]:
        return [self.create_instance(["word"] * count) for count in token_counts]

    def get_batches_stats(self, batches: Iterable[Batch]) -> Dict[str, Union[int, List[int]]]:
        grouped_instances = [batch.instances for batch in batches]
        group_lengths = [len(group) for group in grouped_instances]

        sample_sizes = []
        for batch in batches:
            batch_sequence_length = max(
                instance.get_padding_lengths()["text"]["tokens___tokens"]
                for instance in batch.instances
            )
            sample_sizes.append(batch_sequence_length * len(batch.instances))

        return {
            "batch_lengths": group_lengths,
            "total_instances": sum(group_lengths),
            "sample_sizes": sample_sizes,
        }

    def assert_instances_are_correct(self, candidate_instances):
        # First we need to remove padding tokens from the candidates.

        candidate_instances = [
            tuple(w for w in instance if w != 0) for instance in candidate_instances
        ]
        expected_instances = [
            tuple(instance.fields["text"]._indexed_tokens["tokens"]["tokens"])
            for instance in self.instances
        ]
        assert set(candidate_instances) == set(expected_instances)


class TestBasicIterator(IteratorTest):
    def test_get_num_batches(self):
        # Lazy and instances per epoch not specified.
        assert BasicIterator(batch_size=2).get_num_batches(self.lazy_instances) == 1
        # Lazy and instances per epoch specified.
        assert (
            BasicIterator(batch_size=2, instances_per_epoch=21).get_num_batches(self.lazy_instances)
            == 11
        )
        # Not lazy and instances per epoch specified.
        assert (
            BasicIterator(batch_size=2, instances_per_epoch=21).get_num_batches(self.instances)
            == 11
        )
        # Not lazy and instances per epoch not specified.
        assert BasicIterator(batch_size=2).get_num_batches(self.instances) == 3

    # The BasicIterator should work the same for lazy and non lazy datasets,
    # so each remaining test runs over both.
    def test_yield_one_epoch_iterates_over_the_data_once(self):
        for test_instances in (self.instances, self.lazy_instances):
            iterator = BasicIterator(batch_size=2)
            iterator.index_with(self.vocab)
            batches = list(iterator(test_instances, num_epochs=1))
            # We just want to get the single-token array for the text field in the instance.
            instances = [
                tuple(instance.detach().cpu().numpy())
                for batch in batches
                for instance in batch["text"]["tokens"]["tokens"]
            ]
            assert len(instances) == 5
            self.assert_instances_are_correct(instances)

    def test_call_iterates_over_data_forever(self):
        for test_instances in (self.instances, self.lazy_instances):
            iterator = BasicIterator(batch_size=2)
            iterator.index_with(self.vocab)
            generator = iterator(test_instances)
            batches = [next(generator) for _ in range(18)]  # going over the data 6 times
            # We just want to get the single-token array for the text field in the instance.
            instances = [
                tuple(instance.detach().cpu().numpy())
                for batch in batches
                for instance in batch["text"]["tokens"]["tokens"]
            ]
            assert len(instances) == 5 * 6
            self.assert_instances_are_correct(instances)

    def test_create_batches_groups_correctly(self):

        for test_instances in (self.instances, self.lazy_instances):
            iterator = BasicIterator(batch_size=2)
            batches = list(iterator._create_batches(test_instances, shuffle=False))
            grouped_instances = [batch.instances for batch in batches]
            assert grouped_instances == [
                [self.instances[0], self.instances[1]],
                [self.instances[2], self.instances[3]],
                [self.instances[4]],
            ]

    def test_few_instances_per_epoch(self):

        for test_instances in (self.instances, self.lazy_instances):
            iterator = BasicIterator(batch_size=2, instances_per_epoch=3)
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

    def test_many_instances_per_epoch(self):

        for test_instances in (self.instances, self.lazy_instances):
            iterator = BasicIterator(batch_size=2, instances_per_epoch=7)
            # First epoch: 7 instances -> [2, 2, 2, 1]
            batches = list(iterator._create_batches(test_instances, shuffle=False))
            grouped_instances = [batch.instances for batch in batches]
            assert grouped_instances == [
                [self.instances[0], self.instances[1]],
                [self.instances[2], self.instances[3]],
                [self.instances[4], self.instances[0]],
                [self.instances[1]],
            ]

            # Second epoch: 7 instances -> [2, 2, 2, 1]
            batches = list(iterator._create_batches(test_instances, shuffle=False))
            grouped_instances = [batch.instances for batch in batches]
            assert grouped_instances == [
                [self.instances[2], self.instances[3]],
                [self.instances[4], self.instances[0]],
                [self.instances[1], self.instances[2]],
                [self.instances[3]],
            ]

    def test_epoch_tracking_when_one_epoch_at_a_time(self):
        iterator = BasicIterator(batch_size=2, track_epoch=True)
        iterator.index_with(self.vocab)
        for epoch in range(10):
            for batch in iterator(self.instances, num_epochs=1):
                assert all(epoch_num == epoch for epoch_num in batch["epoch_num"])

    def test_epoch_tracking_multiple_epochs(self):
        iterator = BasicIterator(batch_size=2, track_epoch=True)
        iterator.index_with(self.vocab)

        all_batches = list(iterator(self.instances, num_epochs=10))
        assert len(all_batches) == 10 * 3
        for i, batch in enumerate(all_batches):
            # Should have 3 batches per epoch
            epoch = i // 3
            assert all(epoch_num == epoch for epoch_num in batch["epoch_num"])

    def test_epoch_tracking_forever(self):
        iterator = BasicIterator(batch_size=2, track_epoch=True)
        iterator.index_with(self.vocab)

        it = iterator(self.instances, num_epochs=None)

        all_batches = [next(it) for _ in range(30)]

        assert len(all_batches) == 30
        for i, batch in enumerate(all_batches):
            # Should have 3 batches per epoch
            epoch = i // 3
            assert all(epoch_num == epoch for epoch_num in batch["epoch_num"])

    def test_shuffle(self):
        for test_instances in (self.instances, self.lazy_instances):

            iterator = BasicIterator(batch_size=2, instances_per_epoch=100)

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
            iterator = BasicIterator(batch_size=2, max_instances_in_memory=3)
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
            iterator = BasicIterator(batch_size=1, instances_per_epoch=2)
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
        iterator = BasicIterator.from_params(params)
        assert iterator._batch_size == 32  # default value

        params = Params({"batch_size": 10})
        iterator = BasicIterator.from_params(params)
        assert iterator._batch_size == 10

    def test_maximum_samples_per_batch(self):
        for test_instances in (self.instances, self.lazy_instances):

            iterator = BasicIterator(batch_size=3, maximum_samples_per_batch=["tokens___tokens", 9])
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

        iterator = BasicIterator(batch_size=3, maximum_samples_per_batch=["tokens___tokens", 11])
        iterator.index_with(self.vocab)
        batches = list(iterator._create_batches(test_instances, shuffle=False))
        stats = self.get_batches_stats(batches)

        # ensure all instances are in a batch
        assert stats["total_instances"] == len(token_counts)

        # ensure correct batch sizes
        assert stats["batch_lengths"] == [1, 2]

        # ensure correct sample sizes (<= 11)
        assert stats["sample_sizes"] == [10, 8]
