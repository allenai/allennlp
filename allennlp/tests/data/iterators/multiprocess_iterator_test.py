# pylint: disable=no-self-use,invalid-name
from allennlp.data.iterators import MultiprocessIterator
from allennlp.tests.data.iterators.basic_iterator_test import IteratorTest

class TestMultiprocessIterator(IteratorTest):
    def test_yield_one_epoch_iterates_over_the_data_once(self):
        for test_instances in (self.instances, self.lazy_instances):
            iterator = MultiprocessIterator(num_workers=4, batch_size=2)
            batches = list(iterator(test_instances, num_epochs=1))
            # We just want to get the single-token array for the text field in the instance.
            instances = [tuple(instance.detach().cpu().numpy())
                         for batch in batches
                         for instance in batch['text']["tokens"]]
            assert len(instances) == 5
            self.assert_instances_are_correct(instances)
