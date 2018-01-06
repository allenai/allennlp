# pylint: disable=no-self-use,invalid-name
import pytest
import numpy

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import LazyDataset, Instance, Token, Vocabulary
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer


class TestLazyDataset(AllenNlpTestCase):
    def setUp(self):
        self.vocab = Vocabulary()
        self.vocab.add_token_to_namespace("this")
        self.vocab.add_token_to_namespace("is")
        self.vocab.add_token_to_namespace("a")
        self.vocab.add_token_to_namespace("sentence")
        self.vocab.add_token_to_namespace(".")
        self.token_indexer = {"tokens": SingleIdTokenIndexer()}
        super(TestLazyDataset, self).setUp()

    def test_iterinstances(self):
        dataset = self.get_lazy_dataset()
        dataset.index_instances(self.vocab)
        iterator = iter(dataset)

        instance1 = next(iterator)
        instance2 = next(iterator)

        with pytest.raises(StopIteration):
            _ = next(iterator)

        padding_lengths = instance1.get_padding_lengths()
        tensors1 = instance1.as_tensor_dict(padding_lengths)
        text_1_1 = tensors1['text1']['tokens'].data.cpu().numpy()
        text_1_2 = tensors1['text2']['tokens'].data.cpu().numpy()
        numpy.testing.assert_array_almost_equal(text_1_1, numpy.array([2, 3, 4, 5, 6]))
        numpy.testing.assert_array_almost_equal(text_1_2, numpy.array([2, 3, 4, 1, 5, 6]))

        padding_lengths = instance2.get_padding_lengths()
        tensors2 = instance2.as_tensor_dict(padding_lengths)
        text_2_1 = tensors2['text1']['tokens'].data.cpu().numpy()
        text_2_2 = tensors2['text2']['tokens'].data.cpu().numpy()
        numpy.testing.assert_array_almost_equal(text_2_1, numpy.array([1, 3, 4, 5, 6]))
        numpy.testing.assert_array_almost_equal(text_2_2, numpy.array([2, 3, 1, 0, 0, 0]))

    def get_lazy_dataset(self):
        field1 = TextField([Token(t) for t in ["this", "is", "a", "sentence", "."]],
                           self.token_indexer)
        field2 = TextField([Token(t) for t in ["this", "is", "a", "different", "sentence", "."]],
                           self.token_indexer)
        field3 = TextField([Token(t) for t in ["here", "is", "a", "sentence", "."]],
                           self.token_indexer)
        field4 = TextField([Token(t) for t in ["this", "is", "short"]],
                           self.token_indexer)
        instances = [Instance({"text1": field1, "text2": field2}),
                     Instance({"text1": field3, "text2": field4})]
        return LazyDataset(lambda: (instance for instance in instances))
