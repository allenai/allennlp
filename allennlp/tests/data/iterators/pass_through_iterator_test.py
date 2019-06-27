# pylint: disable=no-self-use,invalid-name
import numpy as np
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Instance, Token, Vocabulary
from allennlp.data.fields import ListField, TextField
from allennlp.data.iterators import BasicIterator, PassThroughIterator
from allennlp.data.iterators.pass_through_iterator import _remove_batch_dim, logger
from allennlp.tests.data.iterators.basic_iterator_test import IteratorTest


def test_remove_batch_dim():
    # Check that first dimension of a tensor is removed
    tensor_with_extra_batch_dim = torch.LongTensor([[1, 2, 3, 4]])
    observed_output = _remove_batch_dim(tensor_with_extra_batch_dim).data.numpy()
    expected_output = np.array([1, 2, 3, 4])
    np.testing.assert_almost_equal(observed_output, expected_output)

    # Check that first dimension of a tensor in a dictionary is removed
    tensor_dict_with_extra_batch_dim = {'tensor': tensor_with_extra_batch_dim}
    observed_output = _remove_batch_dim(tensor_dict_with_extra_batch_dim)
    np.testing.assert_almost_equal(observed_output['tensor'].data.numpy(),
                                   expected_output)

    # Chek that other input types are unaffected
    non_tensor = 'should be ignored'
    assert _remove_batch_dim(non_tensor)

    dict_with_non_tensor = {'non_tensor': non_tensor}
    assert _remove_batch_dim(dict_with_non_tensor) == dict_with_non_tensor


class TestPassThroughIterator(IteratorTest):
    def test_get_num_batches(self):
        # Since batching is assumed to be performed in the DatasetReader, the number of batches
        # (according to the iterator) should always equal the number of instances.
        self.assertEqual(PassThroughIterator().get_num_batches(self.instances),
                         len(self.instances))

    def test_enabling_shuffling_raises_warning(self):
        iterator = PassThroughIterator()
        iterator.index_with(self.vocab)
        generator = iterator(self.instances, shuffle=True)
        with self.assertLogs(logger, level='INFO') as context_manager:
            next(generator)
        self.assertIn('WARNING', context_manager.output[0])

    def test_batch_dim_is_removed(self):
        # Ensure that PassThroughIterator does not add a batch dimension to tensors.

        # First instance is a sequence of four tokens. Thus the expected output is a dict
        # containing a single tensor with shape (4,).
        iterator = PassThroughIterator()
        iterator.index_with(self.vocab)
        generator = iterator(self.instances)
        tensor_dict = next(generator)
        self.assertEqual(tensor_dict['text']['tokens'].size(), (4,))
