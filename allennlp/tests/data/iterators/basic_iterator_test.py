# pylint: disable=no-self-use,invalid-name
from typing import List
from collections import Counter

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Instance, Token, Vocabulary
from allennlp.data.dataset_readers.dataset_reader import _LazyInstances
from allennlp.data.fields import TextField
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer

class IteratorTest(AllenNlpTestCase):
    def setUp(self):
        super(IteratorTest, self).setUp()
        self.token_indexers = {"tokens": SingleIdTokenIndexer()}
        self.vocab = Vocabulary()
        self.this_index = self.vocab.add_token_to_namespace('this')
        self.is_index = self.vocab.add_token_to_namespace('is')
        self.a_index = self.vocab.add_token_to_namespace('a')
        self.sentence_index = self.vocab.add_token_to_namespace('sentence')
        self.another_index = self.vocab.add_token_to_namespace('another')
        self.yet_index = self.vocab.add_token_to_namespace('yet')
        self.very_index = self.vocab.add_token_to_namespace('very')
        self.long_index = self.vocab.add_token_to_namespace('long')
        instances = [
                self.create_instance(["this", "is", "a", "sentence"]),
                self.create_instance(["this", "is", "another", "sentence"]),
                self.create_instance(["yet", "another", "sentence"]),
                self.create_instance(["this", "is", "a", "very", "very", "very", "very", "long", "sentence"]),
                self.create_instance(["sentence"]),
                ]

        class LazyIterable:
            def __iter__(self):
                return (instance for instance in instances)

        self.instances = instances
        self.lazy_instances = LazyIterable()

    def create_instance(self, str_tokens: List[str]):
        tokens = [Token(t) for t in str_tokens]
        instance = Instance({'text': TextField(tokens, self.token_indexers)})
        instance.index_fields(self.vocab)
        return instance

    def assert_instances_are_correct(self, candidate_instances):
        # First we need to remove padding tokens from the candidates.
        # pylint: disable=protected-access
        candidate_instances = [tuple(w for w in instance if w != 0) for instance in candidate_instances]
        expected_instances = [tuple(instance.fields["text"]._indexed_tokens["tokens"])
                              for instance in self.instances]
        assert set(candidate_instances) == set(expected_instances)


class TestBasicIterator(IteratorTest):
    def test_get_num_batches(self):
        # Lazy and instances per epoch not specified.
        assert BasicIterator(batch_size=2).get_num_batches(self.lazy_instances) == 1
        # Lazy and instances per epoch specified.
        assert BasicIterator(batch_size=2, instances_per_epoch=21).get_num_batches(self.lazy_instances) == 11
        # Not lazy and instances per epoch specified.
        assert BasicIterator(batch_size=2, instances_per_epoch=21).get_num_batches(self.instances) == 11
        # Not lazy and instances per epoch not specified.
        assert BasicIterator(batch_size=2).get_num_batches(self.instances) == 3

    # The BasicIterator should work the same for lazy and non lazy datasets,
    # so each remaining test runs over both.
    def test_yield_one_epoch_iterates_over_the_data_once(self):
        for test_instances in (self.instances, self.lazy_instances):
            iterator = BasicIterator(batch_size=2)
            batches = list(iterator(test_instances, num_epochs=1))
            # We just want to get the single-token array for the text field in the instance.
            instances = [tuple(instance.data.cpu().numpy())
                         for batch in batches
                         for instance in batch['text']["tokens"]]
            assert len(instances) == 5
            self.assert_instances_are_correct(instances)

    def test_call_iterates_over_data_forever(self):
        for test_instances in (self.instances, self.lazy_instances):
            generator = BasicIterator(batch_size=2)(test_instances)
            batches = [next(generator) for _ in range(18)]  # going over the data 6 times
            # We just want to get the single-token array for the text field in the instance.
            instances = [tuple(instance.data.cpu().numpy())
                         for batch in batches
                         for instance in batch['text']["tokens"]]
            assert len(instances) == 5 * 6
            self.assert_instances_are_correct(instances)

    def test_create_batches_groups_correctly(self):
        # pylint: disable=protected-access
        for test_instances in (self.instances, self.lazy_instances):
            iterator = BasicIterator(batch_size=2)
            batches = list(iterator._create_batches(test_instances, shuffle=False))
            grouped_instances = [batch.instances for batch in batches]
            assert grouped_instances == [[self.instances[0], self.instances[1]],
                                         [self.instances[2], self.instances[3]],
                                         [self.instances[4]]]

    def test_few_instances_per_epoch(self):
        # pylint: disable=protected-access
        for test_instances in (self.instances, self.lazy_instances):
            iterator = BasicIterator(batch_size=2, instances_per_epoch=3)
            # First epoch: 3 instances -> [2, 1]
            batches = list(iterator._create_batches(test_instances, shuffle=False))
            grouped_instances = [batch.instances for batch in batches]
            assert grouped_instances == [[self.instances[0], self.instances[1]],
                                         [self.instances[2]]]
            # Second epoch: 3 instances -> [2, 1]
            batches = list(iterator._create_batches(test_instances, shuffle=False))
            grouped_instances = [batch.instances for batch in batches]
            assert grouped_instances == [[self.instances[3], self.instances[4]],
                                         [self.instances[0]]]
            # Third epoch: 3 instances -> [2, 1]
            batches = list(iterator._create_batches(test_instances, shuffle=False))
            grouped_instances = [batch.instances for batch in batches]
            assert grouped_instances == [[self.instances[1], self.instances[2]],
                                         [self.instances[3]]]

    def test_many_instances_per_epoch(self):
        # pylint: disable=protected-access
        for test_instances in (self.instances, self.lazy_instances):
            iterator = BasicIterator(batch_size=2, instances_per_epoch=7)
            # First epoch: 7 instances -> [2, 2, 2, 1]
            batches = list(iterator._create_batches(test_instances, shuffle=False))
            grouped_instances = [batch.instances for batch in batches]
            assert grouped_instances == [[self.instances[0], self.instances[1]],
                                         [self.instances[2], self.instances[3]],
                                         [self.instances[4], self.instances[0]],
                                         [self.instances[1]]]

            # Second epoch: 7 instances -> [2, 2, 2, 1]
            batches = list(iterator._create_batches(test_instances, shuffle=False))
            grouped_instances = [batch.instances for batch in batches]
            assert grouped_instances == [[self.instances[2], self.instances[3]],
                                         [self.instances[4], self.instances[0]],
                                         [self.instances[1], self.instances[2]],
                                         [self.instances[3]]]

    def test_shuffle(self):
        # pylint: disable=protected-access
        for test_instances in (self.instances, self.lazy_instances):

            iterator = BasicIterator(batch_size=2, instances_per_epoch=100)

            in_order_batches = list(iterator._create_batches(test_instances, shuffle=False))
            shuffled_batches = list(iterator._create_batches(test_instances, shuffle=True))

            assert len(in_order_batches) == len(shuffled_batches)

            # With 100 instances, shuffling better change the order.
            assert in_order_batches != shuffled_batches

            # But not the counts of the instances.
            in_order_counts = Counter(instance for batch in in_order_batches for instance in batch)
            shuffled_counts = Counter(instance for batch in shuffled_batches for instance in batch)
            assert in_order_counts == shuffled_counts


    def test_max_instances_in_memory(self):
        # pylint: disable=protected-access
        for test_instances in (self.instances, self.lazy_instances):
            iterator = BasicIterator(batch_size=2, max_instances_in_memory=3)
            # One epoch: 5 instances -> [2, 1, 2]
            batches = list(iterator._create_batches(test_instances, shuffle=False))
            grouped_instances = [batch.instances for batch in batches]
            assert grouped_instances == [[self.instances[0], self.instances[1]],
                                         [self.instances[2]],
                                         [self.instances[3], self.instances[4]]]

    def test_multiple_cursors(self):
        # pylint: disable=protected-access
        lazy_instances1 = _LazyInstances(lambda: (i for i in self.instances))
        lazy_instances2 = _LazyInstances(lambda: (i for i in self.instances))

        eager_instances1 = self.instances[:]
        eager_instances2 = self.instances[:]

        for instances1, instances2 in [(eager_instances1, eager_instances2),
                                       (lazy_instances1, lazy_instances2)]:
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
        # pylint: disable=protected-access
        params = Params({})
        iterator = BasicIterator.from_params(params)
        assert iterator._batch_size == 32  # default value

        params = Params({"batch_size": 10})
        iterator = BasicIterator.from_params(params)
        assert iterator._batch_size == 10
