from collections import defaultdict

import pytest
import numpy

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.fields import LabelsField
from allennlp.data.token_indexers import SingleIdTokenIndexer


class TestLabelsField(AllenNlpTestCase):
    def setUp(self):
        super().setUp()

    def test_count_vocab_items_correctly_indexes_tags(self):
        tags = ["B", "I", "O", "O", "O"]
        labels_field = LabelsField(tags, label_namespace="labels")

        counter = defaultdict(lambda: defaultdict(int))
        labels_field.count_vocab_items(counter)

        assert counter["labels"]["B"] == 1
        assert counter["labels"]["I"] == 1
        assert counter["labels"]["O"] == 3
        assert set(counter.keys()) == {"labels"}

    def test_index_converts_field_correctly(self):
        vocab = Vocabulary()
        b_index = vocab.add_token_to_namespace("B", namespace="*labels")
        i_index = vocab.add_token_to_namespace("I", namespace="*labels")
        o_index = vocab.add_token_to_namespace("O", namespace="*labels")

        tags = ["B", "I", "O", "O", "O"]
        labels_field = LabelsField(tags, label_namespace="*labels")
        labels_field.index(vocab)

        assert labels_field._indexed_labels == [b_index, i_index, o_index, o_index, o_index]

    def test_as_tensor_produces_integer_targets(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("B", namespace="*labels")
        vocab.add_token_to_namespace("I", namespace="*labels")
        vocab.add_token_to_namespace("O", namespace="*labels")

        tags = ["B", "I", "O", "O", "O"]
        labels_field = LabelsField(tags, label_namespace="*labels")
        labels_field.index(vocab)
        padding_lengths = labels_field.get_padding_lengths()
        tensor = labels_field.as_tensor(padding_lengths).detach().cpu().numpy()
        numpy.testing.assert_array_almost_equal(tensor, numpy.array([0, 1, 2, 2, 2]))

    def test_labels_field_raises_on_incorrect_type(self):

        with pytest.raises(ConfigurationError):
            _ = LabelsField([[], [], [], [], []])

    def test_class_variables_for_namespace_warnings_work_correctly(self):

        tags = ["B", "I", "O", "O", "O"]
        assert "text" not in LabelsField._already_warned_namespaces
        with self.assertLogs(logger="allennlp.data.fields.labels_field", level="WARNING"):
            _ = LabelsField(tags, label_namespace="text")

        # We've warned once, so we should have set the class variable to False.
        assert "text" in LabelsField._already_warned_namespaces
        with pytest.raises(AssertionError):
            with self.assertLogs(logger="allennlp.data.fields.labels_field", level="WARNING"):
                _ = LabelsField(tags, label_namespace="text")

        # ... but a new namespace should still log a warning.
        assert "text2" not in LabelsField._already_warned_namespaces
        with self.assertLogs(logger="allennlp.data.fields.labels_field", level="WARNING"):
            _ = LabelsField(tags, label_namespace="text2")

    def test_printing_doesnt_crash(self):
        tags = ["B", "I", "O", "O", "O"]
        labels_field = LabelsField(tags, label_namespace="labels")
        print(labels_field)

    def test_sequence_methods(self):
        tags = ["B", "I", "O", "O", "O"]
        labels_field = LabelsField(tags, label_namespace="labels")

        assert len(labels_field) == 5
        assert labels_field[1] == "I"
        assert [label for label in labels_field] == tags
