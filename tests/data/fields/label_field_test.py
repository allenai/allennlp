# pylint: disable=no-self-use,invalid-name
import numpy

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields import LabelField
from allennlp.common.testing import AllenNlpTestCase


class TestLabelField(AllenNlpTestCase):

    def test_pad_returns_integer_array(self):
        label = LabelField(5, num_labels=10)
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
