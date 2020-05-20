import logging

import numpy
import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.fields import MultiLabelField
from allennlp.data.vocabulary import Vocabulary


class TestMultiLabelField(AllenNlpTestCase):
    def test_as_tensor_returns_integer_tensor(self):
        f = MultiLabelField([2, 3], skip_indexing=True, label_namespace="test1", num_labels=5)
        tensor = f.as_tensor(f.get_padding_lengths()).detach().cpu().tolist()
        assert tensor == [0, 0, 1, 1, 0]
        assert {type(item) for item in tensor} == {int}

    def test_multilabel_field_can_index_with_vocab(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("rel0", namespace="rel_labels")
        vocab.add_token_to_namespace("rel1", namespace="rel_labels")
        vocab.add_token_to_namespace("rel2", namespace="rel_labels")

        f = MultiLabelField(["rel1", "rel0"], label_namespace="rel_labels")
        f.index(vocab)
        tensor = f.as_tensor(f.get_padding_lengths()).detach().cpu().numpy()
        numpy.testing.assert_array_almost_equal(tensor, numpy.array([1, 1, 0]))

    def test_multilabel_field_raises_with_non_integer_labels_and_no_indexing(self):
        with pytest.raises(ConfigurationError):
            _ = MultiLabelField(["non integer field"], skip_indexing=True)

    def test_multilabel_field_raises_with_no_indexing_and_missing_num_labels(self):
        with pytest.raises(ConfigurationError):
            _ = MultiLabelField([0, 2], skip_indexing=True, num_labels=None)

    def test_multilabel_field_raises_with_no_indexing_and_wrong_num_labels(self):
        with pytest.raises(ConfigurationError):
            _ = MultiLabelField([0, 2, 4], skip_indexing=True, num_labels=3)

    def test_multilabel_field_raises_with_incorrect_label_type(self):
        with pytest.raises(ConfigurationError):
            _ = MultiLabelField([1, 2], skip_indexing=False)

    def test_multilabel_field_raises_with_given_num_labels(self):
        with pytest.raises(ConfigurationError):
            _ = MultiLabelField([1, 2], skip_indexing=False, num_labels=4)

    def test_multilabel_field_empty_field_works(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("label1", namespace="test_empty_labels")
        vocab.add_token_to_namespace("label2", namespace="test_empty_labels")

        f = MultiLabelField([], label_namespace="test_empty_labels")
        f.index(vocab)
        tensor = f.as_tensor(f.get_padding_lengths()).detach().cpu().numpy()
        numpy.testing.assert_array_almost_equal(tensor, numpy.array([0, 0]))
        g = f.empty_field()
        g.index(vocab)
        tensor = g.as_tensor(g.get_padding_lengths()).detach().cpu().numpy()
        numpy.testing.assert_array_almost_equal(tensor, numpy.array([0, 0]))

        h = MultiLabelField(
            [0, 0, 1], label_namespace="test_empty_labels", num_labels=3, skip_indexing=True
        )
        tensor = h.empty_field().as_tensor(None).detach().cpu().numpy()
        numpy.testing.assert_array_almost_equal(tensor, numpy.array([0, 0, 0]))

    def test_class_variables_for_namespace_warnings_work_correctly(self, caplog):
        with caplog.at_level(logging.WARNING, logger="allennlp.data.fields.multilabel_field"):
            assert "text" not in MultiLabelField._already_warned_namespaces
            _ = MultiLabelField(["test"], label_namespace="text")
            assert caplog.records

            # We've warned once, so we should have set the class variable to False.
            assert "text" in MultiLabelField._already_warned_namespaces
            caplog.clear()
            _ = MultiLabelField(["test2"], label_namespace="text")
            assert not caplog.records

            # ... but a new namespace should still log a warning.
            assert "text2" not in MultiLabelField._already_warned_namespaces
            caplog.clear()
            _ = MultiLabelField(["test"], label_namespace="text2")
            assert caplog

    def test_printing_doesnt_crash(self):
        field = MultiLabelField(["label"], label_namespace="namespace")
        print(field)
