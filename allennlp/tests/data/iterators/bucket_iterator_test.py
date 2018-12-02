# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data.iterators import BucketIterator
from allennlp.tests.data.iterators.basic_iterator_test import IteratorTest


class TestBucketIterator(IteratorTest):
    # pylint: disable=protected-access
    def test_create_batches_groups_correctly(self):
        iterator = BucketIterator(batch_size=2, padding_noise=0, sorting_keys=[('text', 'num_tokens')])
        iterator.index_with(self.vocab)
        batches = list(iterator._create_batches(self.instances, shuffle=False))
        grouped_instances = [batch.instances for batch in batches]
        assert grouped_instances == [[self.instances[4], self.instances[2]],
                                     [self.instances[0], self.instances[1]],
                                     [self.instances[3]]]

    def test_create_batches_groups_correctly_with_max_instances(self):
        # If we knew all the instances, the correct order is 4 -> 2 -> 0 -> 1 -> 3.
        # Here max_instances_in_memory is 3, so we load instances [0, 1, 2]
        # and then bucket them by size into batches of size 2 to get [2, 0] -> [1].
        # Then we load the remaining instances and bucket them by size to get [4, 3].
        iterator = BucketIterator(batch_size=2,
                                  padding_noise=0,
                                  sorting_keys=[('text', 'num_tokens')],
                                  max_instances_in_memory=3)
        iterator.index_with(self.vocab)
        for test_instances in (self.instances, self.lazy_instances):
            batches = list(iterator._create_batches(test_instances, shuffle=False))
            grouped_instances = [batch.instances for batch in batches]
            assert grouped_instances == [[self.instances[2], self.instances[0]],
                                         [self.instances[1]],
                                         [self.instances[4], self.instances[3]]]

    def test_biggest_batch_first_works(self):
        iterator = BucketIterator(batch_size=2,
                                  padding_noise=0,
                                  sorting_keys=[('text', 'num_tokens')],
                                  biggest_batch_first=True)
        iterator.index_with(self.vocab)
        batches = list(iterator._create_batches(self.instances, shuffle=False))
        grouped_instances = [batch.instances for batch in batches]
        assert grouped_instances == [[self.instances[3]],
                                     [self.instances[0], self.instances[1]],
                                     [self.instances[4], self.instances[2]]]

    def test_from_params(self):
        # pylint: disable=protected-access
        params = Params({})

        with pytest.raises(ConfigurationError):
            iterator = BucketIterator.from_params(params)

        sorting_keys = [("s1", "nt"), ("s2", "nt2")]
        params['sorting_keys'] = sorting_keys
        iterator = BucketIterator.from_params(params)

        assert iterator._sorting_keys == sorting_keys
        assert iterator._padding_noise == 0.1
        assert not iterator._biggest_batch_first
        assert iterator._batch_size == 32

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

    def test_bucket_iterator_maximum_samples_per_batch(self):
        iterator = BucketIterator(
                batch_size=3,
                padding_noise=0,
                sorting_keys=[('text', 'num_tokens')],
                maximum_samples_per_batch=['num_tokens', 9]
        )
        iterator.index_with(self.vocab)
        batches = list(iterator._create_batches(self.instances, shuffle=False))
        stats = self.get_batches_stats(batches)

        # ensure all instances are in a batch
        assert stats['total_instances'] == len(self.instances)

        # ensure correct batch sizes
        assert stats['batch_lengths'] == [2, 2, 1]

        # ensure correct sample sizes (<= 9)
        assert stats['sample_sizes'] == [6, 8, 9]

    def test_maximum_samples_per_batch_packs_tightly(self):
        token_counts = [10, 4, 3]
        test_instances = self.create_instances_from_token_counts(token_counts)

        iterator = BucketIterator(
                batch_size=3,
                padding_noise=0,
                sorting_keys=[('text', 'num_tokens')],
                maximum_samples_per_batch=['num_tokens', 11]
        )
        iterator.index_with(self.vocab)
        batches = list(iterator._create_batches(test_instances, shuffle=False))
        stats = self.get_batches_stats(batches)

        # ensure all instances are in a batch
        assert stats['total_instances'] == len(test_instances)

        # ensure correct batch sizes
        assert stats['batch_lengths'] == [2, 1]

        # ensure correct sample sizes (<= 11)
        assert stats['sample_sizes'] == [8, 10]
