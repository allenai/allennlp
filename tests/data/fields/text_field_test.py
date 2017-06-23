# pylint: disable=no-self-use,invalid-name
from collections import defaultdict

import pytest
import numpy

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import token_indexers

from allennlp.testing.test_case import DeepQaTestCase
from allennlp.common.checks import ConfigurationError


class TestTextField(DeepQaTestCase):
    def setUp(self):
        self.vocab = Vocabulary()
        self.vocab.add_token_to_namespace("sentence", namespace='words')
        self.vocab.add_token_to_namespace("A", namespace='words')
        self.vocab.add_token_to_namespace("A", namespace='characters')
        self.vocab.add_token_to_namespace("s", namespace='characters')
        self.vocab.add_token_to_namespace("e", namespace='characters')
        self.vocab.add_token_to_namespace("n", namespace='characters')
        self.vocab.add_token_to_namespace("t", namespace='characters')
        self.vocab.add_token_to_namespace("c", namespace='characters')
        super(TestTextField, self).setUp()

    def test_field_counts_vocab_items_correctly(self):
        field = TextField(["This", "is", "a", "sentence", "."],
                          token_indexers=[token_indexers["single id"]("words")])
        namespace_token_counts = defaultdict(lambda: defaultdict(int))
        field.count_vocab_items(namespace_token_counts)

        assert namespace_token_counts["words"]["This"] == 1
        assert namespace_token_counts["words"]["is"] == 1
        assert namespace_token_counts["words"]["a"] == 1
        assert namespace_token_counts["words"]["sentence"] == 1
        assert namespace_token_counts["words"]["."] == 1
        assert list(namespace_token_counts.keys()) == ["words"]

        field = TextField(["This", "is", "a", "sentence", "."],
                          token_indexers=[token_indexers["characters"]("characters")])
        namespace_token_counts = defaultdict(lambda: defaultdict(int))
        field.count_vocab_items(namespace_token_counts)

        assert namespace_token_counts["characters"]["T"] == 1
        assert namespace_token_counts["characters"]["h"] == 1
        assert namespace_token_counts["characters"]["i"] == 2
        assert namespace_token_counts["characters"]["s"] == 3
        assert namespace_token_counts["characters"]["a"] == 1
        assert namespace_token_counts["characters"]["e"] == 3
        assert namespace_token_counts["characters"]["n"] == 2
        assert namespace_token_counts["characters"]["t"] == 1
        assert namespace_token_counts["characters"]["c"] == 1
        assert namespace_token_counts["characters"]["."] == 1
        assert list(namespace_token_counts.keys()) == ["characters"]

        field = TextField(["This", "is", "a", "sentence", "."],
                          token_indexers=[token_indexers["single id"]("words"),
                                          token_indexers["characters"]("characters")])
        namespace_token_counts = defaultdict(lambda: defaultdict(int))
        field.count_vocab_items(namespace_token_counts)
        assert namespace_token_counts["characters"]["T"] == 1
        assert namespace_token_counts["characters"]["h"] == 1
        assert namespace_token_counts["characters"]["i"] == 2
        assert namespace_token_counts["characters"]["s"] == 3
        assert namespace_token_counts["characters"]["a"] == 1
        assert namespace_token_counts["characters"]["e"] == 3
        assert namespace_token_counts["characters"]["n"] == 2
        assert namespace_token_counts["characters"]["t"] == 1
        assert namespace_token_counts["characters"]["c"] == 1
        assert namespace_token_counts["characters"]["."] == 1
        assert namespace_token_counts["words"]["This"] == 1
        assert namespace_token_counts["words"]["is"] == 1
        assert namespace_token_counts["words"]["a"] == 1
        assert namespace_token_counts["words"]["sentence"] == 1
        assert namespace_token_counts["words"]["."] == 1
        assert set(namespace_token_counts.keys()) == {"words", "characters"}

    def test_index_converts_field_correctly(self):
        vocab = Vocabulary()
        sentence_index = vocab.add_token_to_namespace("sentence", namespace='words')
        capital_a_index = vocab.add_token_to_namespace("A", namespace='words')
        capital_a_char_index = vocab.add_token_to_namespace("A", namespace='characters')
        s_index = vocab.add_token_to_namespace("s", namespace='characters')
        e_index = vocab.add_token_to_namespace("e", namespace='characters')
        n_index = vocab.add_token_to_namespace("n", namespace='characters')
        t_index = vocab.add_token_to_namespace("t", namespace='characters')
        c_index = vocab.add_token_to_namespace("c", namespace='characters')

        field = TextField(["A", "sentence"], [token_indexers["single id"](token_namespace="words")])
        field.index(vocab)
        # pylint: disable=protected-access
        assert field._indexed_tokens == [[capital_a_index, sentence_index]]

        field1 = TextField(["A", "sentence"], [token_indexers["characters"](character_namespace="characters")])
        field1.index(vocab)
        assert field1._indexed_tokens == [[[capital_a_char_index],
                                           [s_index, e_index, n_index, t_index,
                                            e_index, n_index, c_index, e_index]]]
        field2 = TextField(["A", "sentence"],
                           token_indexers=[token_indexers["single id"](token_namespace="words"),
                                           token_indexers["characters"](character_namespace="characters")])
        field2.index(vocab)
        assert field2._indexed_tokens == [[capital_a_index, sentence_index],
                                          [[capital_a_char_index],
                                           [s_index, e_index, n_index, t_index,
                                            e_index, n_index, c_index, e_index]]]
        # pylint: enable=protected-access

    def test_get_padding_lengths_raises_if_no_indexed_tokens(self):

        field = TextField(["This", "is", "a", "sentence", "."],
                          token_indexers=[token_indexers["single id"]("words")])
        with pytest.raises(ConfigurationError):
            field.get_padding_lengths()

    def test_padding_lengths_are_computed_correctly(self):
        field = TextField(["This", "is", "a", "sentence", "."],
                          token_indexers=[token_indexers["single id"]("words")])
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        assert padding_lengths == {"num_tokens": 5}

        field = TextField(["This", "is", "a", "sentence", "."],
                          token_indexers=[token_indexers["characters"]("characters")])
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        assert padding_lengths == {"num_tokens": 5, "num_token_characters": 8}

        field = TextField(["This", "is", "a", "sentence", "."],
                          token_indexers=[token_indexers["characters"]("characters"),
                                          token_indexers["single id"]("words")])
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        assert padding_lengths == {"num_tokens": 5, "num_token_characters": 8}

    def test_pad_handles_words(self):
        field = TextField(["This", "is", "a", "sentence", "."],
                          token_indexers=[token_indexers["single id"]("words")])
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        arrays = field.pad(padding_lengths)
        numpy.testing.assert_array_almost_equal(arrays[0], numpy.array([1, 1, 1, 2, 1]))

    def test_pad_handles_longer_lengths(self):
        field = TextField(["This", "is", "a", "sentence", "."],
                          token_indexers=[token_indexers["single id"]("words")])
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        padding_lengths["num_tokens"] = 10
        arrays = field.pad(padding_lengths)
        numpy.testing.assert_array_almost_equal(arrays[0], numpy.array([1, 1, 1, 2, 1, 0, 0, 0, 0, 0]))

    def test_pad_handles_characters(self):
        field = TextField(["This", "is", "a", "sentence", "."],
                          token_indexers=[token_indexers["characters"]("characters")])
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        arrays = field.pad(padding_lengths)
        expected_character_array = numpy.array([[1, 1, 1, 3, 0, 0, 0, 0],
                                                [1, 3, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 0, 0, 0, 0, 0, 0],
                                                [3, 4, 5, 6, 4, 5, 7, 4],
                                                [1, 0, 0, 0, 0, 0, 0, 0]])
        numpy.testing.assert_array_almost_equal(arrays[0], expected_character_array)

    def test_pad_handles_words_and_characters_with_longer_lengths(self):
        field = TextField(["a", "sentence", "."],
                          token_indexers=[token_indexers["single id"]("words"),
                                          token_indexers["characters"]("characters")])
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        padding_lengths["num_tokens"] = 5
        padding_lengths["num_token_characters"] = 10
        arrays = field.pad(padding_lengths)

        numpy.testing.assert_array_almost_equal(arrays[0], numpy.array([1, 2, 1, 0, 0]))
        numpy.testing.assert_array_almost_equal(arrays[1], numpy.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                        [3, 4, 5, 6, 4, 5, 7, 4, 0, 0],
                                                                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
