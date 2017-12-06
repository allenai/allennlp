# pylint: disable=no-self-use,invalid-name
import numpy
import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.fields import LabelField
from allennlp.data.vocabulary import Vocabulary


class TestLabelField(AllenNlpTestCase):
    def test_as_tensor_returns_integer_tensor(self):
        label = LabelField(5, skip_indexing=True)
        tensor = label.as_tensor(label.get_padding_lengths()).data.cpu().numpy()
        numpy.testing.assert_array_almost_equal(tensor, numpy.array([5]))

    def test_label_field_can_index_with_vocab(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("entailment", namespace="labels")
        vocab.add_token_to_namespace("contradiction", namespace="labels")
        vocab.add_token_to_namespace("neutral", namespace="labels")

        label = LabelField("entailment")
        label.index(vocab)
        tensor = label.as_tensor(label.get_padding_lengths()).data.cpu().numpy()
        numpy.testing.assert_array_almost_equal(tensor, numpy.array([0]))

    def test_label_field_raises_with_non_integer_labels_and_no_indexing(self):
        with pytest.raises(ConfigurationError):
            _ = LabelField("non integer field", skip_indexing=True)

    def test_label_field_raises_with_incorrect_label_type(self):
        with pytest.raises(ConfigurationError):
            _ = LabelField([], skip_indexing=False)

    def test_label_field_empty_field_works(self):
        label = LabelField("test")
        empty_label = label.empty_field()
        assert empty_label.label == -1

    def test_class_variables_for_namespace_warnings_work_correctly(self):
        # pylint: disable=protected-access
        assert "text" not in LabelField._already_warned_namespaces
        with self.assertLogs(logger="allennlp.data.fields.label_field", level="WARNING"):
            _ = LabelField("test", label_namespace="text")

        # We've warned once, so we should have set the class variable to False.
        assert "text" in LabelField._already_warned_namespaces
        with pytest.raises(AssertionError):
            with self.assertLogs(logger="allennlp.data.fields.label_field", level="WARNING"):
                _ = LabelField("test2", label_namespace="text")

        # ... but a new namespace should still log a warning.
        assert "text2" not in LabelField._already_warned_namespaces
        with self.assertLogs(logger="allennlp.data.fields.label_field", level="WARNING"):
            _ = LabelField("test", label_namespace="text2")
