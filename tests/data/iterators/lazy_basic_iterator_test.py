# pylint: disable=no-self-use,invalid-name
from typing import List

from allennlp.common import Params
from allennlp.data import Instance, Token
from allennlp.data.dataset import LazyDataset
from allennlp.data.fields import TextField
from allennlp.data.iterators.lazy_basic_iterator import LazyBasicIterator
from tests.data.iterators.basic_iterator_test import IteratorTest

class LazyBasicIteratorTest(IteratorTest):
    def setUp(self):
        super(LazyBasicIteratorTest, self).setUp()
        self.dataset = LazyDataset(lambda: iter(self.instances))

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


class TestLazyIterator(LazyBasicIteratorTest):
    def test_yield_one_epoch_iterates_over_the_data_once(self):
        iterator = LazyBasicIterator(batch_size=2)
        batches = list(iterator(self.dataset, num_epochs=1))
        # We just want to get the single-token array for the text field in the instance.
        instances = [tuple(instance.data.cpu().numpy())
                     for batch in batches
                     for instance in batch['text']["tokens"]]
        assert len(instances) == 5
        self.assert_instances_are_correct(instances)

    def test_call_iterates_over_data_forever(self):
        generator = LazyBasicIterator(batch_size=2)(self.dataset)
        batches = [next(generator) for _ in range(18)]  # going over the data 6 times
        # We just want to get the single-token array for the text field in the instance.
        instances = [tuple(instance.data.cpu().numpy())
                     for batch in batches
                     for instance in batch['text']["tokens"]]
        assert len(instances) == 5 * 6
        self.assert_instances_are_correct(instances)

    def test_create_batches_groups_correctly(self):
        # pylint: disable=protected-access
        iterator = LazyBasicIterator(batch_size=2)
        batches = list(iterator._create_batches(self.dataset, shuffle=False))
        grouped_instances = [batch.instances for batch in batches]
        assert grouped_instances == [[self.instances[0], self.instances[1]],
                                     [self.instances[2], self.instances[3]],
                                     [self.instances[4]]]

    def test_small_epochs(self):
        # pylint: disable=protected-access
        iterator = LazyBasicIterator(batch_size=2, instances_per_epoch=2)

        # We should loop around when we get to the end
        batches = list(iterator._create_batches(self.dataset, shuffle=False))
        grouped_instances = [batch.instances for batch in batches]
        assert grouped_instances == [[self.instances[0], self.instances[1]]]

        batches = list(iterator._create_batches(self.dataset, shuffle=False))
        grouped_instances = [batch.instances for batch in batches]
        assert grouped_instances == [[self.instances[2], self.instances[3]]]

        batches = list(iterator._create_batches(self.dataset, shuffle=False))
        grouped_instances = [batch.instances for batch in batches]
        assert grouped_instances == [[self.instances[4], self.instances[0]]]

        batches = list(iterator._create_batches(self.dataset, shuffle=False))
        grouped_instances = [batch.instances for batch in batches]
        assert grouped_instances == [[self.instances[1], self.instances[2]]]

        batches = list(iterator._create_batches(self.dataset, shuffle=False))
        grouped_instances = [batch.instances for batch in batches]
        assert grouped_instances == [[self.instances[3], self.instances[4]]]

        batches = list(iterator._create_batches(self.dataset, shuffle=False))
        grouped_instances = [batch.instances for batch in batches]
        assert grouped_instances == [[self.instances[0], self.instances[1]]]

    def test_multiple_cursors(self):
        # pylint: disable=protected-access
        dataset1 = LazyDataset(lambda: iter(self.instances))
        dataset1.index_instances(self.vocab)

        dataset2 = LazyDataset(lambda: iter(self.instances))
        dataset2.index_instances(self.vocab)

        iterator = LazyBasicIterator(batch_size=1, instances_per_epoch=2)

        # First epoch through dataset1
        batches = list(iterator._create_batches(dataset1, shuffle=False))
        grouped_instances = [batch.instances for batch in batches]
        assert grouped_instances == [[self.instances[0]], [self.instances[1]]]

        # First epoch through dataset2
        batches = list(iterator._create_batches(dataset2, shuffle=False))
        grouped_instances = [batch.instances for batch in batches]
        assert grouped_instances == [[self.instances[0]], [self.instances[1]]]

        # Second epoch through dataset1
        batches = list(iterator._create_batches(dataset1, shuffle=False))
        grouped_instances = [batch.instances for batch in batches]
        assert grouped_instances == [[self.instances[2]], [self.instances[3]]]

        # Second epoch through dataset2
        batches = list(iterator._create_batches(dataset2, shuffle=False))
        grouped_instances = [batch.instances for batch in batches]
        assert grouped_instances == [[self.instances[2]], [self.instances[3]]]


    def test_from_params(self):
        # pylint: disable=protected-access
        params = Params({})
        iterator = LazyBasicIterator.from_params(params)
        assert iterator._batch_size == 32  # default value

        params = Params({"batch_size": 10})
        iterator = LazyBasicIterator.from_params(params)
        assert iterator._batch_size == 10
