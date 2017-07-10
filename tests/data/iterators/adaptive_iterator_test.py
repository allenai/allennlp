# pylint: disable=no-self-use,invalid-name
from allennlp.data.iterators import AdaptiveIterator
from tests.data.iterators.basic_iterator_test import IteratorTest


class TestAdaptiveIterator(IteratorTest):
    # pylint: disable=protected-access
    def test_create_batches_groups_correctly(self):
        iterator = AdaptiveIterator(adaptive_memory_usage_constant=12,
                                    padding_memory_scaling=lambda x: x['text']['num_tokens'],
                                    padding_noise=0,
                                    sorting_keys=[('text', 'num_tokens')])
        grouped_instances = iterator._create_batches(self.dataset, shuffle=False)
        assert grouped_instances == [[self.instances[4], self.instances[2], self.instances[0]],
                                     [self.instances[1]],
                                     [self.instances[3]]]

    def test_create_batches_respects_maximum_batch_size(self):
        iterator = AdaptiveIterator(adaptive_memory_usage_constant=12,
                                    padding_memory_scaling=lambda x: x['text']['num_tokens'],
                                    maximum_batch_size=2,
                                    padding_noise=0,
                                    sorting_keys=[('text', 'num_tokens')])
        grouped_instances = iterator._create_batches(self.dataset, shuffle=False)
        assert grouped_instances == [[self.instances[4], self.instances[2]],
                                     [self.instances[0], self.instances[1]],
                                     [self.instances[3]]]

    def test_biggest_batch_first_passes_off_to_bucket_iterator(self):
        iterator = AdaptiveIterator(adaptive_memory_usage_constant=8,
                                    padding_memory_scaling=lambda x: x['text']['num_tokens'],
                                    padding_noise=0,
                                    sorting_keys=[('text', 'num_tokens')],
                                    biggest_batch_first=True,
                                    batch_size=2)
        grouped_instances = iterator._create_batches(self.dataset, shuffle=False)
        assert grouped_instances == [[self.instances[3]],
                                     [self.instances[0], self.instances[1]],
                                     [self.instances[4], self.instances[2]]]
