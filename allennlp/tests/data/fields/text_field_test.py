# pylint: disable=no-self-use,invalid-name
from collections import defaultdict
from typing import Dict, List

import pytest
import numpy

from allennlp.data import Token, Vocabulary
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer, TokenIndexer

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import pad_sequence_to_length


class DictReturningTokenIndexer(TokenIndexer):
    """
    A stub TokenIndexer that returns multiple arrays of different lengths.
    """
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        pass

    def tokens_to_indices(self, tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]: # pylint: disable=unused-argument
        return {
                "token_ids": [10, 15] + \
                         [vocabulary.get_token_index(token.text, 'words') for token in tokens] + \
                         [25],
                "additional_key": [22, 29]
        }

    def get_padding_token(self) -> int:
        return 0

    def get_padding_lengths(self, token: int) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    def pad_token_sequence(self,
                           tokens: Dict[str, List[int]],
                           desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, List[int]]:  # pylint: disable=unused-argument
        return {key: pad_sequence_to_length(val, desired_num_tokens[key]) for key, val in tokens.items()}

    def get_keys(self, index_name: str) -> List[str]:
        # pylint: disable=unused-argument,no-self-use
        return ["token_ids", "additional_key"]


class TestTextField(AllenNlpTestCase):
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
        field = TextField([Token(t) for t in ["This", "is", "a", "sentence", "."]],
                          token_indexers={"words": SingleIdTokenIndexer("words")})
        namespace_token_counts = defaultdict(lambda: defaultdict(int))
        field.count_vocab_items(namespace_token_counts)

        assert namespace_token_counts["words"]["This"] == 1
        assert namespace_token_counts["words"]["is"] == 1
        assert namespace_token_counts["words"]["a"] == 1
        assert namespace_token_counts["words"]["sentence"] == 1
        assert namespace_token_counts["words"]["."] == 1
        assert list(namespace_token_counts.keys()) == ["words"]

        field = TextField([Token(t) for t in ["This", "is", "a", "sentence", "."]],
                          token_indexers={"characters": TokenCharactersIndexer("characters",
                                                                               min_padding_length=1)})
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

        field = TextField([Token(t) for t in ["This", "is", "a", "sentence", "."]],
                          token_indexers={"words": SingleIdTokenIndexer("words"),
                                          "characters": TokenCharactersIndexer("characters",
                                                                               min_padding_length=1)})
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

        field = TextField([Token(t) for t in ["A", "sentence"]],
                          {"words": SingleIdTokenIndexer(namespace="words")})
        field.index(vocab)
        # pylint: disable=protected-access
        assert field._indexed_tokens["words"] == [capital_a_index, sentence_index]

        field1 = TextField([Token(t) for t in ["A", "sentence"]],
                           {"characters": TokenCharactersIndexer(namespace="characters",
                                                                 min_padding_length=1)})
        field1.index(vocab)
        assert field1._indexed_tokens["characters"] == [[capital_a_char_index],
                                                        [s_index, e_index, n_index, t_index,
                                                         e_index, n_index, c_index, e_index]]
        field2 = TextField([Token(t) for t in ["A", "sentence"]],
                           token_indexers={"words": SingleIdTokenIndexer(namespace="words"),
                                           "characters": TokenCharactersIndexer(namespace="characters",
                                                                                min_padding_length=1)})
        field2.index(vocab)
        assert field2._indexed_tokens["words"] == [capital_a_index, sentence_index]
        assert field2._indexed_tokens["characters"] == [[capital_a_char_index],
                                                        [s_index, e_index, n_index, t_index,
                                                         e_index, n_index, c_index, e_index]]
        # pylint: enable=protected-access

    def test_get_padding_lengths_raises_if_no_indexed_tokens(self):

        field = TextField([Token(t) for t in ["This", "is", "a", "sentence", "."]],
                          token_indexers={"words": SingleIdTokenIndexer("words")})
        with pytest.raises(ConfigurationError):
            field.get_padding_lengths()

    def test_padding_lengths_are_computed_correctly(self):
        field = TextField([Token(t) for t in ["This", "is", "a", "sentence", "."]],
                          token_indexers={"words": SingleIdTokenIndexer("words")})
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        assert padding_lengths == {"words_length": 5, "num_tokens": 5}

        field = TextField([Token(t) for t in ["This", "is", "a", "sentence", "."]],
                          token_indexers={"characters": TokenCharactersIndexer("characters",
                                                                               min_padding_length=1)})
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        assert padding_lengths == {"num_tokens": 5, "characters_length": 5, "num_token_characters": 8}

        field = TextField([Token(t) for t in ["This", "is", "a", "sentence", "."]],
                          token_indexers={"characters": TokenCharactersIndexer("characters",
                                                                               min_padding_length=1),
                                          "words": SingleIdTokenIndexer("words")})
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        assert padding_lengths == {"num_tokens": 5,
                                   "characters_length": 5,
                                   "words_length": 5,
                                   "num_token_characters": 8}

    def test_as_tensor_handles_words(self):
        field = TextField([Token(t) for t in ["This", "is", "a", "sentence", "."]],
                          token_indexers={"words": SingleIdTokenIndexer("words")})
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        tensor_dict = field.as_tensor(padding_lengths)
        numpy.testing.assert_array_almost_equal(tensor_dict["words"].detach().cpu().numpy(),
                                                numpy.array([1, 1, 1, 2, 1]))

    def test_as_tensor_handles_longer_lengths(self):
        field = TextField([Token(t) for t in ["This", "is", "a", "sentence", "."]],
                          token_indexers={"words": SingleIdTokenIndexer("words")})
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        padding_lengths["words_length"] = 10
        tensor_dict = field.as_tensor(padding_lengths)
        numpy.testing.assert_array_almost_equal(tensor_dict["words"].detach().cpu().numpy(),
                                                numpy.array([1, 1, 1, 2, 1, 0, 0, 0, 0, 0]))

    def test_as_tensor_handles_characters(self):
        field = TextField([Token(t) for t in ["This", "is", "a", "sentence", "."]],
                          token_indexers={"characters": TokenCharactersIndexer("characters",
                                                                               min_padding_length=1)})
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        tensor_dict = field.as_tensor(padding_lengths)
        expected_character_array = numpy.array([[1, 1, 1, 3, 0, 0, 0, 0],
                                                [1, 3, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 0, 0, 0, 0, 0, 0],
                                                [3, 4, 5, 6, 4, 5, 7, 4],
                                                [1, 0, 0, 0, 0, 0, 0, 0]])
        numpy.testing.assert_array_almost_equal(tensor_dict["characters"].detach().cpu().numpy(),
                                                expected_character_array)

    def test_as_tensor_handles_words_and_characters_with_longer_lengths(self):
        field = TextField([Token(t) for t in ["a", "sentence", "."]],
                          token_indexers={"words": SingleIdTokenIndexer("words"),
                                          "characters": TokenCharactersIndexer("characters",
                                                                               min_padding_length=1)})
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        padding_lengths["words_length"] = 5
        padding_lengths["characters_length"] = 5
        padding_lengths["num_token_characters"] = 10
        tensor_dict = field.as_tensor(padding_lengths)

        numpy.testing.assert_array_almost_equal(tensor_dict["words"].detach().cpu().numpy(),
                                                numpy.array([1, 2, 1, 0, 0]))
        numpy.testing.assert_array_almost_equal(tensor_dict["characters"].detach().cpu().numpy(),
                                                numpy.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                             [3, 4, 5, 6, 4, 5, 7, 4, 0, 0],
                                                             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

    def test_printing_doesnt_crash(self):
        field = TextField([Token(t) for t in ["A", "sentence"]],
                          {"words": SingleIdTokenIndexer(namespace="words")})
        print(field)

    def test_token_indexer_returns_dict(self):
        field = TextField([Token(t) for t in ["A", "sentence"]],
                          token_indexers={"field_with_dict": DictReturningTokenIndexer(),
                                          "words": SingleIdTokenIndexer("words"),
                                          "characters": TokenCharactersIndexer("characters",
                                                                               min_padding_length=1)})
        field.index(self.vocab)
        padding_lengths = field.get_padding_lengths()
        assert padding_lengths == {
                'token_ids_length': 5,
                'additional_key_length': 2,
                'words_length': 2,
                'characters_length': 2,
                'num_token_characters': 8,
                'num_tokens': 5,
        }
        padding_lengths['token_ids_length'] = 7
        padding_lengths['additional_key_length'] = 3
        padding_lengths['words_length'] = 4
        padding_lengths['characters_length'] = 4
        tensors = field.as_tensor(padding_lengths)
        assert list(tensors['token_ids'].shape) == [7]
        assert list(tensors['additional_key'].shape) == [3]
        assert list(tensors['words'].shape) == [4]
        assert list(tensors['characters'].shape) == [4, 8]

    def test_sequence_methods(self):
        field = TextField([Token(t) for t in ["This", "is", "a", "sentence", "."]], {})

        assert len(field) == 5
        assert field[1].text == "is"
        assert [token.text for token in field] == ["This", "is", "a", "sentence", "."]
