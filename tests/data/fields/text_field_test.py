from collections import defaultdict
from typing import Dict, List

import numpy
import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer, TokenIndexer


class DictReturningTokenIndexer(TokenIndexer):
    """
    A stub TokenIndexer that returns multiple arrays of different lengths.
    """

    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        pass

    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary
    ) -> Dict[str, List[int]]:
        return {
            "token_ids": (
                [10, 15]
                + [vocabulary.get_token_index(token.text, "words") for token in tokens]
                + [25]
            ),
            "additional_key": [22, 29],
        }


class TestTextField(AllenNlpTestCase):
    def setup_method(self):
        self.vocab = Vocabulary()
        self.vocab.add_token_to_namespace("sentence", namespace="words")
        self.vocab.add_token_to_namespace("A", namespace="words")
        self.vocab.add_token_to_namespace("A", namespace="characters")
        self.vocab.add_token_to_namespace("s", namespace="characters")
        self.vocab.add_token_to_namespace("e", namespace="characters")
        self.vocab.add_token_to_namespace("n", namespace="characters")
        self.vocab.add_token_to_namespace("t", namespace="characters")
        self.vocab.add_token_to_namespace("c", namespace="characters")
        super().setup_method()

    def test_field_counts_vocab_items_correctly(self):
        field = TextField(
            [Token(t) for t in ["This", "is", "a", "sentence", "."]],
            token_indexers={"words": SingleIdTokenIndexer("words")},
        )
        namespace_token_counts = defaultdict(lambda: defaultdict(int))
        field.count_vocab_items(namespace_token_counts)

        assert namespace_token_counts["words"]["This"] == 1
        assert namespace_token_counts["words"]["is"] == 1
        assert namespace_token_counts["words"]["a"] == 1
        assert namespace_token_counts["words"]["sentence"] == 1
        assert namespace_token_counts["words"]["."] == 1
        assert list(namespace_token_counts.keys()) == ["words"]

        field = TextField(
            [Token(t) for t in ["This", "is", "a", "sentence", "."]],
            token_indexers={
                "characters": TokenCharactersIndexer("characters", min_padding_length=1)
            },
        )
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

        field = TextField(
            [Token(t) for t in ["This", "is", "a", "sentence", "."]],
            token_indexers={
                "words": SingleIdTokenIndexer("words"),
                "characters": TokenCharactersIndexer("characters", min_padding_length=1),
            },
        )
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
        sentence_index = vocab.add_token_to_namespace("sentence", namespace="words")
        capital_a_index = vocab.add_token_to_namespace("A", namespace="words")
        capital_a_char_index = vocab.add_token_to_namespace("A", namespace="characters")
        s_index = vocab.add_token_to_namespace("s", namespace="characters")
        e_index = vocab.add_token_to_namespace("e", namespace="characters")
        n_index = vocab.add_token_to_namespace("n", namespace="characters")
        t_index = vocab.add_token_to_namespace("t", namespace="characters")
        c_index = vocab.add_token_to_namespace("c", namespace="characters")

        field = TextField(
            [Token(t) for t in ["A", "sentence"]],
            {"words": SingleIdTokenIndexer(namespace="words")},
        )
        field.index(vocab)

        assert field._indexed_tokens["words"]["tokens"] == [capital_a_index, sentence_index]

        field1 = TextField(
            [Token(t) for t in ["A", "sentence"]],
            {"characters": TokenCharactersIndexer(namespace="characters", min_padding_length=1)},
        )
        field1.index(vocab)
        assert field1._indexed_tokens["characters"]["token_characters"] == [
            [capital_a_char_index],
            [s_index, e_index, n_index, t_index, e_index, n_index, c_index, e_index],
        ]
        field2 = TextField(
            [Token(t) for t in ["A", "sentence"]],
            token_indexers={
                "words": SingleIdTokenIndexer(namespace="words"),
                "characters": TokenCharactersIndexer(namespace="characters", min_padding_length=1),
            },
        )
        field2.index(vocab)
        assert field2._indexed_tokens["words"]["tokens"] == [capital_a_index, sentence_index]
        assert field2._indexed_tokens["characters"]["token_characters"] == [
            [capital_a_char_index],
            [s_index, e_index, n_index, t_index, e_index, n_index, c_index, e_index],
        ]

    def test_get_padding_lengths_raises_if_no_indexed_tokens(self):

        field = TextField(
            [Token(t) for t in ["This", "is", "a", "sentence", "."]],
            token_indexers={"words": SingleIdTokenIndexer("words")},
        )
        with pytest.raises(ConfigurationError):
            field.get_padding_lengths()

    def test_padding_lengths_are_computed_correctly(self):
        field = TextField(
            [Token(t) for t in ["This", "is", "a", "sentence", "."]],
            token_indexers={"words": SingleIdTokenIndexer("words")},
        )
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        assert padding_lengths == {"words___tokens": 5}

        field = TextField(
            [Token(t) for t in ["This", "is", "a", "sentence", "."]],
            token_indexers={
                "characters": TokenCharactersIndexer("characters", min_padding_length=1)
            },
        )
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        assert padding_lengths == {
            "characters___token_characters": 5,
            "characters___num_token_characters": 8,
        }

        field = TextField(
            [Token(t) for t in ["This", "is", "a", "sentence", "."]],
            token_indexers={
                "characters": TokenCharactersIndexer("characters", min_padding_length=1),
                "words": SingleIdTokenIndexer("words"),
            },
        )
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        assert padding_lengths == {
            "characters___token_characters": 5,
            "characters___num_token_characters": 8,
            "words___tokens": 5,
        }

    def test_as_tensor_handles_words(self):
        field = TextField(
            [Token(t) for t in ["This", "is", "a", "sentence", "."]],
            token_indexers={"words": SingleIdTokenIndexer("words")},
        )
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        tensor_dict = field.as_tensor(padding_lengths)
        numpy.testing.assert_array_almost_equal(
            tensor_dict["words"]["tokens"].detach().cpu().numpy(), numpy.array([1, 1, 1, 2, 1])
        )

    def test_as_tensor_handles_longer_lengths(self):
        field = TextField(
            [Token(t) for t in ["This", "is", "a", "sentence", "."]],
            token_indexers={"words": SingleIdTokenIndexer("words")},
        )
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        padding_lengths["words___tokens"] = 10
        tensor_dict = field.as_tensor(padding_lengths)
        numpy.testing.assert_array_almost_equal(
            tensor_dict["words"]["tokens"].detach().cpu().numpy(),
            numpy.array([1, 1, 1, 2, 1, 0, 0, 0, 0, 0]),
        )

    def test_as_tensor_handles_characters(self):
        field = TextField(
            [Token(t) for t in ["This", "is", "a", "sentence", "."]],
            token_indexers={
                "characters": TokenCharactersIndexer("characters", min_padding_length=1)
            },
        )
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        tensor_dict = field.as_tensor(padding_lengths)
        expected_character_array = numpy.array(
            [
                [1, 1, 1, 3, 0, 0, 0, 0],
                [1, 3, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [3, 4, 5, 6, 4, 5, 7, 4],
                [1, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        numpy.testing.assert_array_almost_equal(
            tensor_dict["characters"]["token_characters"].detach().cpu().numpy(),
            expected_character_array,
        )

    def test_as_tensor_handles_characters_if_empty_field(self):
        field = TextField(
            [],
            token_indexers={
                "characters": TokenCharactersIndexer("characters", min_padding_length=1)
            },
        )
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        tensor_dict = field.as_tensor(padding_lengths)
        expected_character_array = numpy.array([])
        numpy.testing.assert_array_almost_equal(
            tensor_dict["characters"]["token_characters"].detach().cpu().numpy(),
            expected_character_array,
        )

    def test_as_tensor_handles_words_and_characters_with_longer_lengths(self):
        field = TextField(
            [Token(t) for t in ["a", "sentence", "."]],
            token_indexers={
                "words": SingleIdTokenIndexer("words"),
                "characters": TokenCharactersIndexer("characters", min_padding_length=1),
            },
        )
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        padding_lengths["words___tokens"] = 5
        padding_lengths["characters___token_characters"] = 5
        padding_lengths["characters___num_token_characters"] = 10
        tensor_dict = field.as_tensor(padding_lengths)

        numpy.testing.assert_array_almost_equal(
            tensor_dict["words"]["tokens"].detach().cpu().numpy(), numpy.array([1, 2, 1, 0, 0])
        )
        numpy.testing.assert_array_almost_equal(
            tensor_dict["characters"]["token_characters"].detach().cpu().numpy(),
            numpy.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [3, 4, 5, 6, 4, 5, 7, 4, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        )

    def test_printing_doesnt_crash(self):
        field = TextField(
            [Token(t) for t in ["A", "sentence"]],
            {"words": SingleIdTokenIndexer(namespace="words")},
        )
        print(field)

    def test_token_indexer_returns_dict(self):
        field = TextField(
            [Token(t) for t in ["A", "sentence"]],
            token_indexers={
                "field_with_dict": DictReturningTokenIndexer(),
                "words": SingleIdTokenIndexer("words"),
                "characters": TokenCharactersIndexer("characters", min_padding_length=1),
            },
        )
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        assert padding_lengths == {
            "field_with_dict___token_ids": 5,
            "field_with_dict___additional_key": 2,
            "words___tokens": 2,
            "characters___token_characters": 2,
            "characters___num_token_characters": 8,
        }
        padding_lengths["field_with_dict___token_ids"] = 7
        padding_lengths["field_with_dict___additional_key"] = 3
        padding_lengths["words___tokens"] = 4
        padding_lengths["characters___token_characters"] = 4
        tensors = field.as_tensor(padding_lengths)
        assert list(tensors["field_with_dict"]["token_ids"].shape) == [7]
        assert list(tensors["field_with_dict"]["additional_key"].shape) == [3]
        assert list(tensors["words"]["tokens"].shape) == [4]
        assert list(tensors["characters"]["token_characters"].shape) == [4, 8]

    def test_token_padding_lengths_are_computed_correctly(self):
        field = TextField(
            [Token(t) for t in ["A", "sentence"]],
            token_indexers={
                "field_with_dict": DictReturningTokenIndexer(token_min_padding_length=3),
                "words": SingleIdTokenIndexer("words", token_min_padding_length=3),
                "characters": TokenCharactersIndexer(
                    "characters", min_padding_length=1, token_min_padding_length=3
                ),
            },
        )
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        assert padding_lengths == {
            "field_with_dict___token_ids": 5,
            "field_with_dict___additional_key": 3,
            "words___tokens": 3,
            "characters___token_characters": 3,
            "characters___num_token_characters": 8,
        }
        tensors = field.as_tensor(padding_lengths)
        assert tensors["field_with_dict"]["additional_key"].tolist()[-1] == 0
        assert tensors["words"]["tokens"].tolist()[-1] == 0
        assert tensors["characters"]["token_characters"].tolist()[-1] == [0] * 8

    def test_sequence_methods(self):
        field = TextField([Token(t) for t in ["This", "is", "a", "sentence", "."]], {})

        assert len(field) == 5
        assert field[1].text == "is"
        assert [token.text for token in field] == ["This", "is", "a", "sentence", "."]
