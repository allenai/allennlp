# pylint: disable=no-self-use,invalid-name
import numpy
import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token
from allennlp.data.fields import TextField, IndexField
from allennlp.data.token_indexers import SingleIdTokenIndexer

class TestIndexField(AllenNlpTestCase):
    def setUp(self):
        super(TestIndexField, self).setUp()
        self.text = TextField([Token(t) for t in ["here", "is", "a", "sentence", "."]],
                              {"words": SingleIdTokenIndexer("words")})

    def test_as_tensor_converts_field_correctly(self):
        index_field = IndexField(4, self.text)
        tensor = index_field.as_tensor(index_field.get_padding_lengths()).data.cpu().numpy()
        numpy.testing.assert_array_equal(tensor, numpy.array([4]))

    def test_index_field_raises_on_incorrect_label_type(self):
        with pytest.raises(ConfigurationError):
            _ = IndexField("hello", self.text)

    def test_index_field_empty_field_works(self):
        index_field = IndexField(4, self.text)
        empty_index = index_field.empty_field()
        assert empty_index.sequence_index == -1
