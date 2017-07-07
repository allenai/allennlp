# pylint: disable=no-self-use,invalid-name
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
