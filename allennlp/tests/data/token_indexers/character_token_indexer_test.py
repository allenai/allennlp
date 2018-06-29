# pylint: disable=no-self-use,invalid-name
from collections import defaultdict

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.token_indexers import TokenCharactersIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer


class CharacterTokenIndexerTest(AllenNlpTestCase):
    def test_count_vocab_items_respects_casing(self):
        indexer = TokenCharactersIndexer("characters")
        counter = defaultdict(lambda: defaultdict(int))
        indexer.count_vocab_items(Token("Hello"), counter)
        indexer.count_vocab_items(Token("hello"), counter)
        assert counter["characters"] == {"h": 1, "H": 1, "e": 2, "l": 4, "o": 2}

        indexer = TokenCharactersIndexer("characters", CharacterTokenizer(lowercase_characters=True))
        counter = defaultdict(lambda: defaultdict(int))
        indexer.count_vocab_items(Token("Hello"), counter)
        indexer.count_vocab_items(Token("hello"), counter)
        assert counter["characters"] == {"h": 2, "e": 2, "l": 4, "o": 2}

    def test_as_array_produces_token_sequence(self):
        indexer = TokenCharactersIndexer("characters")
        padded_tokens = indexer.pad_token_sequence([[1, 2, 3, 4, 5], [1, 2, 3], [1]],
                                                   desired_num_tokens=4,
                                                   padding_lengths={"num_token_characters": 10})
        assert padded_tokens == [[1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
                                 [1, 2, 3, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    def test_token_to_indices_produces_correct_characters(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("A", namespace='characters')
        vocab.add_token_to_namespace("s", namespace='characters')
        vocab.add_token_to_namespace("e", namespace='characters')
        vocab.add_token_to_namespace("n", namespace='characters')
        vocab.add_token_to_namespace("t", namespace='characters')
        vocab.add_token_to_namespace("c", namespace='characters')

        indexer = TokenCharactersIndexer("characters")
        indices = indexer.token_to_indices(Token("sentential"), vocab)
        assert indices == [3, 4, 5, 6, 4, 5, 6, 1, 1, 1]
