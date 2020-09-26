import pytest
import numpy

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Instance, Token, Vocabulary
from allennlp.data.batch import Batch
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer


class TestDataset(AllenNlpTestCase):
    def setup_method(self):
        self.vocab = Vocabulary()
        self.vocab.add_token_to_namespace("this")
        self.vocab.add_token_to_namespace("is")
        self.vocab.add_token_to_namespace("a")
        self.vocab.add_token_to_namespace("sentence")
        self.vocab.add_token_to_namespace(".")
        self.token_indexer = {"tokens": SingleIdTokenIndexer()}
        self.instances = self.get_instances()
        super().setup_method()

    def test_instances_must_have_homogeneous_fields(self):
        instance1 = Instance({"tag": (LabelField(1, skip_indexing=True))})
        instance2 = Instance({"words": TextField([Token("hello")], {})})
        with pytest.raises(ConfigurationError):
            _ = Batch([instance1, instance2])

    def test_padding_lengths_uses_max_instance_lengths(self):
        dataset = Batch(self.instances)
        dataset.index_instances(self.vocab)
        padding_lengths = dataset.get_padding_lengths()
        assert padding_lengths == {"text1": {"tokens___tokens": 5}, "text2": {"tokens___tokens": 6}}

    def test_as_tensor_dict(self):
        dataset = Batch(self.instances)
        dataset.index_instances(self.vocab)
        padding_lengths = dataset.get_padding_lengths()
        tensors = dataset.as_tensor_dict(padding_lengths)
        text1 = tensors["text1"]["tokens"]["tokens"].detach().cpu().numpy()
        text2 = tensors["text2"]["tokens"]["tokens"].detach().cpu().numpy()

        numpy.testing.assert_array_almost_equal(
            text1, numpy.array([[2, 3, 4, 5, 6], [1, 3, 4, 5, 6]])
        )
        numpy.testing.assert_array_almost_equal(
            text2, numpy.array([[2, 3, 4, 1, 5, 6], [2, 3, 1, 0, 0, 0]])
        )

    def get_instances(self):
        field1 = TextField(
            [Token(t) for t in ["this", "is", "a", "sentence", "."]], self.token_indexer
        )
        field2 = TextField(
            [Token(t) for t in ["this", "is", "a", "different", "sentence", "."]],
            self.token_indexer,
        )
        field3 = TextField(
            [Token(t) for t in ["here", "is", "a", "sentence", "."]], self.token_indexer
        )
        field4 = TextField([Token(t) for t in ["this", "is", "short"]], self.token_indexer)
        instances = [
            Instance({"text1": field1, "text2": field2}),
            Instance({"text1": field3, "text2": field4}),
        ]
        return instances
