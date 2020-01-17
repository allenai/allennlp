from allennlp.data.iterators.pass_through_iterator import PassThroughIterator, logger
from allennlp.tests.data.iterators.basic_iterator_test import IteratorTest


class TestPassThroughIterator(IteratorTest):
    def test_get_num_batches(self):
        # Since batching is assumed to be performed in the DatasetReader, the number of batches
        # (according to the iterator) should always equal the number of instances.
        self.assertEqual(PassThroughIterator().get_num_batches(self.instances), len(self.instances))

    def test_enabling_shuffling_raises_warning(self):
        iterator = PassThroughIterator()
        iterator.index_with(self.vocab)
        generator = iterator(self.instances, shuffle=True)
        with self.assertLogs(logger, level="INFO") as context_manager:
            next(generator)
        self.assertIn("WARNING", context_manager.output[0])

    def test_batch_dim_is_removed(self):
        # Ensure that PassThroughIterator does not add a batch dimension to tensors.

        # First instance is a sequence of four tokens. Thus the expected output is a dict
        # containing a single tensor with shape (4,).
        iterator = PassThroughIterator()
        iterator.index_with(self.vocab)
        generator = iterator(self.instances)
        tensor_dict = next(generator)
        self.assertEqual(tensor_dict["text"]["tokens"]["tokens"].size(), (4,))
