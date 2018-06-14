# pylint: disable=no-self-use,invalid-name
from pytest import raises

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data.iterators import LMIterator
from allennlp.tests.data.iterators.basic_iterator_test import IteratorTest


class TestLMIterator(IteratorTest):
    # pylint: disable=protected-access

    def test_get_num_batches(self):
        assert LMIterator(batch_size=2).get_num_batches(self.instances) == 2

    def test_create_batches(self):

        iterator = LMIterator(batch_size=2)
        batches = list(iterator._create_batches(self.instances, shuffle=False))
        grouped_instances = [batch.instances for batch in batches]

        assert grouped_instances == [[self.instances[0], self.instances[2]],
                                     [self.instances[1], self.instances[3]]]

    def test_from_params(self):
        # pylint: disable=protected-access

        param_dict = {
                "batch_size": 100,
                "lazy": False
        }

        iterator = LMIterator.from_params(Params(param_dict))
        assert iterator._batch_size == 100
        assert iterator._lazy == False
