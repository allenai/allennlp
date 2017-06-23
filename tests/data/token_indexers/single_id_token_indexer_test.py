# pylint: disable=no-self-use,invalid-name
from collections import defaultdict

from allennlp.data.token_indexers import token_indexers
from allennlp.testing.test_case import DeepQaTestCase


class TestSingleIdTokenIndexer(DeepQaTestCase):

    def test_count_vocab_items_respects_casing(self):
        indexer = token_indexers["single id"]("*words")
        counter = defaultdict(lambda: defaultdict(int))
        indexer.count_vocab_items("Hello", counter)
        indexer.count_vocab_items("hello", counter)
        assert counter["*words"] == {"hello": 1, "Hello": 1}

        indexer = token_indexers["single id"]("*words", lowercase_tokens=True)
        counter = defaultdict(lambda: defaultdict(int))
        indexer.count_vocab_items("Hello", counter)
        indexer.count_vocab_items("hello", counter)
        assert counter["*words"] == {"hello": 2}

    def test_pad_token_sequence(self):
        indexer = token_indexers["single id"]("*words")
        padded_tokens = indexer.pad_token_sequence([1, 2, 3, 4, 5], 10, {})
        assert padded_tokens == [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]
