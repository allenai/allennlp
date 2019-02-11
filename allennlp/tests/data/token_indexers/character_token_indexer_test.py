# pylint: disable=no-self-use,invalid-name
from collections import defaultdict

import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.token_indexers import TokenCharactersIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer


class CharacterTokenIndexerTest(AllenNlpTestCase):
    def test_count_vocab_items_respects_casing(self):
        indexer = TokenCharactersIndexer("characters", min_padding_length=5)
        counter = defaultdict(lambda: defaultdict(int))
        indexer.count_vocab_items(Token("Hello"), counter)
        indexer.count_vocab_items(Token("hello"), counter)
        assert counter["characters"] == {"h": 1, "H": 1, "e": 2, "l": 4, "o": 2}

        indexer = TokenCharactersIndexer("characters", CharacterTokenizer(lowercase_characters=True),
                                         min_padding_length=5)
        counter = defaultdict(lambda: defaultdict(int))
        indexer.count_vocab_items(Token("Hello"), counter)
        indexer.count_vocab_items(Token("hello"), counter)
        assert counter["characters"] == {"h": 2, "e": 2, "l": 4, "o": 2}

    def test_as_array_produces_token_sequence(self):
        indexer = TokenCharactersIndexer("characters", min_padding_length=1)
        padded_tokens = indexer.pad_token_sequence({'k': [[1, 2, 3, 4, 5], [1, 2, 3], [1]]},
                                                   desired_num_tokens={'k': 4},
                                                   padding_lengths={"num_token_characters": 10})
        assert padded_tokens == {'k': [[1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
                                       [1, 2, 3, 0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}

    def test_tokens_to_indices_produces_correct_characters(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("A", namespace='characters')
        vocab.add_token_to_namespace("s", namespace='characters')
        vocab.add_token_to_namespace("e", namespace='characters')
        vocab.add_token_to_namespace("n", namespace='characters')
        vocab.add_token_to_namespace("t", namespace='characters')
        vocab.add_token_to_namespace("c", namespace='characters')

        indexer = TokenCharactersIndexer("characters", min_padding_length=1)
        indices = indexer.tokens_to_indices([Token("sentential")], vocab, "char")
        assert indices == {"char": [[3, 4, 5, 6, 4, 5, 6, 1, 1, 1]]}

    def test_start_and_end_tokens(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("A", namespace='characters')  # 2
        vocab.add_token_to_namespace("s", namespace='characters')  # 3
        vocab.add_token_to_namespace("e", namespace='characters')  # 4
        vocab.add_token_to_namespace("n", namespace='characters')  # 5
        vocab.add_token_to_namespace("t", namespace='characters')  # 6
        vocab.add_token_to_namespace("c", namespace='characters')  # 7
        vocab.add_token_to_namespace("<", namespace='characters')  # 8
        vocab.add_token_to_namespace(">", namespace='characters')  # 9
        vocab.add_token_to_namespace("/", namespace='characters')  # 10

        indexer = TokenCharactersIndexer("characters", start_tokens=["<s>"], end_tokens=["</s>"],
                                         min_padding_length=1)
        indices = indexer.tokens_to_indices([Token("sentential")], vocab, "char")
        assert indices == {"char": [[8, 3, 9],
                                    [3, 4, 5, 6, 4, 5, 6, 1, 1, 1],
                                    [8, 10, 3, 9]]}

    def test_min_padding_length(self):
        sentence = "AllenNLP is awesome ."
        tokens = [Token(token) for token in sentence.split(" ")]
        vocab = Vocabulary()
        vocab.add_token_to_namespace("A", namespace="characters")  # 2
        vocab.add_token_to_namespace("l", namespace="characters")  # 3
        vocab.add_token_to_namespace("e", namespace="characters")  # 4
        vocab.add_token_to_namespace("n", namespace="characters")  # 5
        vocab.add_token_to_namespace("N", namespace="characters")  # 6
        vocab.add_token_to_namespace("L", namespace="characters")  # 7
        vocab.add_token_to_namespace("P", namespace="characters")  # 8
        vocab.add_token_to_namespace("i", namespace="characters")  # 9
        vocab.add_token_to_namespace("s", namespace="characters")  # 10
        vocab.add_token_to_namespace("a", namespace="characters")  # 11
        vocab.add_token_to_namespace("w", namespace="characters")  # 12
        vocab.add_token_to_namespace("o", namespace="characters")  # 13
        vocab.add_token_to_namespace("m", namespace="characters")  # 14
        vocab.add_token_to_namespace(".", namespace="characters")  # 15

        indexer = TokenCharactersIndexer("characters", min_padding_length=10)
        indices = indexer.tokens_to_indices(tokens, vocab, "char")
        key_padding_lengths = "num_token_characters"
        value_padding_lengths = 0
        for token in indices["char"]:
            item = indexer.get_padding_lengths(token)
            value = item.values()
            value_padding_lengths = max(value_padding_lengths, max(value))
        padded = indexer.pad_token_sequence(indices,
                                            {"char": len(indices["char"])},
                                            {key_padding_lengths: value_padding_lengths})
        assert padded == {"char": [[2, 3, 3, 4, 5, 6, 7, 8, 0, 0],
                                   [9, 10, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [11, 12, 4, 10, 13, 14, 4, 0, 0, 0],
                                   [15, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}

    def test_warn_min_padding_length(self):
        with pytest.warns(UserWarning, match=r"using the default value \(0\) of `min_padding_length`"):
            TokenCharactersIndexer("characters")
