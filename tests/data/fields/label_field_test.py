# pylint: disable=no-self-use,invalid-name
import numpy
import pytest

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields import LabelField
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.checks import ConfigurationError


class TestLabelField(AllenNlpTestCase):

    def test_as_array_returns_integer_array(self):
        label = LabelField(5, skip_indexing=True)
        array = label.as_array(label.get_padding_lengths())
        numpy.testing.assert_array_almost_equal(array, numpy.array([5]))

    def test_label_field_can_index_with_vocab(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("entailment", namespace="labels")
        vocab.add_token_to_namespace("contradiction", namespace="labels")
        vocab.add_token_to_namespace("neutral", namespace="labels")

        label = LabelField("entailment")
        label.index(vocab)
        array = label.as_array(label.get_padding_lengths())
        numpy.testing.assert_array_almost_equal(array, numpy.array([0]))

    def test_label_field_raises_with_non_integer_labels_and_no_indexing(self):
        with pytest.raises(ConfigurationError):
            _ = LabelField("non integer field", skip_indexing=True)

    def test_label_field_raises_with_incorrect_label_type(self):
        with pytest.raises(ConfigurationError):
            _ = LabelField([], skip_indexing=False)
