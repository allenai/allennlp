# pylint: disable=no-self-use,invalid-name
from pytest import raises

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data.iterators import AdaptiveIterator
from tests.data.iterators.basic_iterator_test import IteratorTest


class TestAdaptiveIterator(IteratorTest):
    # pylint: disable=protected-access
    def test_create_batches_groups_correctly(self):
        iterator = AdaptiveIterator(adaptive_memory_usage_constant=12,
                                    padding_memory_scaling=lambda x: x['text']['num_tokens'],
                                    padding_noise=0,
                                    sorting_keys=[('text', 'num_tokens')])
        batches = list(iterator._create_batches(self.instances, shuffle=False))
        grouped_instances = [batch.instances for batch in batches]
        assert grouped_instances == [[self.instances[4], self.instances[2], self.instances[0]],
                                     [self.instances[1]],
                                     [self.instances[3]]]

    def test_create_batches_respects_maximum_batch_size(self):
        iterator = AdaptiveIterator(adaptive_memory_usage_constant=12,
                                    padding_memory_scaling=lambda x: x['text']['num_tokens'],
                                    maximum_batch_size=2,
                                    padding_noise=0,
                                    sorting_keys=[('text', 'num_tokens')])
        batches = list(iterator._create_batches(self.instances, shuffle=False))
        grouped_instances = [batch.instances for batch in batches]
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
        batches = list(iterator._create_batches(self.instances, shuffle=False))
        grouped_instances = [batch.instances for batch in batches]
        assert grouped_instances == [[self.instances[3]],
                                     [self.instances[0], self.instances[1]],
                                     [self.instances[4], self.instances[2]]]

    def test_from_params(self):
        # pylint: disable=protected-access
        params = Params({})
        # not all params have default values
        with raises(ConfigurationError):
            _ = AdaptiveIterator.from_params(params)

        param_dict = {
                "adaptive_memory_usage_constant": 10,
                "padding_memory_scaling": lambda x: 2.4,
                "sorting_keys": ["tokens"]
        }

        iterator = AdaptiveIterator.from_params(Params(param_dict))
        assert iterator._adaptive_memory_usage_constant == 10
        assert iterator._padding_memory_scaling({}) == 2.4
        assert iterator._maximum_batch_size == 10000
