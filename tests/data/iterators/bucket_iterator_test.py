# pylint: disable=no-self-use,invalid-name
from allennlp.common import Params
from allennlp.data.iterators import BucketIterator
from tests.data.iterators.basic_iterator_test import IteratorTest


class TestBucketIterator(IteratorTest):
    # pylint: disable=protected-access
    def test_create_batches_groups_correctly(self):
        iterator = BucketIterator(batch_size=2, padding_noise=0, sorting_keys=[('text', 'num_tokens')])
        grouped_instances = iterator._create_batches(self.dataset, shuffle=False)
        assert grouped_instances == [[self.instances[4], self.instances[2]],
                                     [self.instances[0], self.instances[1]],
                                     [self.instances[3]]]

    def test_biggest_batch_first_works(self):
        iterator = BucketIterator(batch_size=2,
                                  padding_noise=0,
                                  sorting_keys=[('text', 'num_tokens')],
                                  biggest_batch_first=True)
        grouped_instances = iterator._create_batches(self.dataset, shuffle=False)
        assert grouped_instances == [[self.instances[3]],
                                     [self.instances[0], self.instances[1]],
                                     [self.instances[4], self.instances[2]]]

    def test_from_params(self):
        # pylint: disable=protected-access
        params = Params({})
        iterator = BucketIterator.from_params(params)
        assert iterator._sorting_keys == []
        assert iterator._padding_noise == 0.1
        assert not iterator._biggest_batch_first
        assert iterator._batch_size == 32

        sorting_keys = [("s1", "nt"), ("s2", "nt2")]
        params = Params({
                "sorting_keys": sorting_keys,
                "padding_noise": 0.5,
                "biggest_batch_first": True,
                "batch_size": 100
        })

        iterator = BucketIterator.from_params(params)
        assert iterator._sorting_keys == sorting_keys
        assert iterator._padding_noise == 0.5
        assert iterator._biggest_batch_first
        assert iterator._batch_size == 100
