from collections import defaultdict

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer


class TestSingleIdTokenIndexer(AllenNlpTestCase):
    def test_count_vocab_items_respects_casing(self):
        indexer = SingleIdTokenIndexer("words")
        counter = defaultdict(lambda: defaultdict(int))
        indexer.count_vocab_items(Token("Hello"), counter)
        indexer.count_vocab_items(Token("hello"), counter)
        assert counter["words"] == {"hello": 1, "Hello": 1}

        indexer = SingleIdTokenIndexer("words", lowercase_tokens=True)
        counter = defaultdict(lambda: defaultdict(int))
        indexer.count_vocab_items(Token("Hello"), counter)
        indexer.count_vocab_items(Token("hello"), counter)
        assert counter["words"] == {"hello": 2}

    def test_as_array_produces_token_sequence(self):
        indexer = SingleIdTokenIndexer("words")
        padded_tokens = indexer.as_padded_tensor_dict({"tokens": [1, 2, 3, 4, 5]}, {"tokens": 10})
        assert padded_tokens["tokens"].tolist() == [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]

    def test_count_other_features(self):
        indexer = SingleIdTokenIndexer("other_features", feature_name="is_bold")
        counter = defaultdict(lambda: defaultdict(int))
        token = Token("Header")
        token.is_bold = "True"
        indexer.count_vocab_items(token, counter)
        assert counter["other_features"] == {"True": 1}
