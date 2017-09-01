# pylint: disable=no-self-use,invalid-name
import numpy
import pytest
from allennlp.data.fields import TextField, IndexField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.checks import ConfigurationError

class TestIndexField(AllenNlpTestCase):

    def setUp(self):
        super(TestIndexField, self).setUp()
        self.text = TextField(["here", "is", "a", "sentence", "."],
                              {"words": SingleIdTokenIndexer("words")})

    def test_index_field_inherits_padding_lengths_from_text_field(self):

        index_field = IndexField(4, self.text)
        assert index_field.get_padding_lengths() == {"num_options": 5}

    def test_as_array_converts_field_correctly(self):
        index_field = IndexField(4, self.text)
        array = index_field.as_array(index_field.get_padding_lengths())
        numpy.testing.assert_array_equal(array, numpy.array([4]))

    def test_index_field_raises_on_incorrect_label_type(self):
        with pytest.raises(ConfigurationError):
            _ = IndexField("hello", self.text)
