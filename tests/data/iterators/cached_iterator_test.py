# pylint: disable=no-self-use,invalid-name

from allennlp.common.checks import ConfigurationError
from allennlp.data.iterators import CachedIterator
from tests.data.iterators.basic_iterator_test import IteratorTest


class TestCachedIterator(IteratorTest):
    # pylint: disable=protected-access
    def test_configuration_error_with_max_instances_in_memory(self):
        try:
            CachedIterator(batch_size=2, padding_noise=0, sorting_keys=[('text', 'num_tokens')],
                           max_instances_in_memory=3)
        except ConfigurationError:
            return
        assert False

    def test_configuration_error_with_lazy(self):
        try:
            iterator = CachedIterator(batch_size=2, padding_noise=0, sorting_keys=[('text', 'num_tokens')])
            [x for x in iterator(self.lazy_instances, num_epochs=2)] # pylint: disable=expression-not-assigned
            assert False
        except ConfigurationError:
            return
        assert False

    def test_caching(self):
        iterator = CachedIterator(batch_size=2, padding_noise=0, sorting_keys=[('text', 'num_tokens')])
        assert not iterator.cached_batches

        batches = [x for x in iterator(self.instances, shuffle=False, num_epochs=2)]
        assert len(iterator.cached_batches) == 1
        assert id(batches[0]) == id(batches[3])
        assert id(batches[1]) == id(batches[4])
        assert id(batches[2]) == id(batches[5])

        batches2 = [x for x in iterator(self.instances, shuffle=False, num_epochs=1)]
        assert len(iterator.cached_batches) == 1
        assert id(batches[0]) == id(batches2[0])
        assert id(batches[1]) == id(batches2[1])
        assert id(batches[2]) == id(batches2[2])

        batches3 = [x for x in iterator(self.instances[:], shuffle=False, num_epochs=1)]
        assert len(iterator.cached_batches) == 2
        assert id(batches[0]) != id(batches3[0])
        assert id(batches[1]) != id(batches3[1])
        assert id(batches[2]) != id(batches3[2])

        batches4 = [x for x in iterator(self.instances, shuffle=True, num_epochs=2)]
        ids4 = [id(x) for x in batches4]
        assert ids4[:3] != ids4[3:]
        ids1 = [id(x) for x in batches]
        assert sorted(ids4) == sorted(ids1)
