from collections import defaultdict
from dataclasses import dataclass

import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer


@dataclass(init=False)
class TokenWithStyle(Token):
    __slots__ = ["is_bold"]

    is_bold: bool

    def __init__(self, text: str = None, is_bold: bool = False):
        super().__init__(text=text)
        self.is_bold = is_bold


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
        token = TokenWithStyle("Header")
        token.is_bold = "True"
        indexer.count_vocab_items(token, counter)
        assert counter["other_features"] == {"True": 1}

    def test_count_vocab_items_with_non_default_feature_name(self):
        tokenizer = SpacyTokenizer(parse=True)
        tokens = tokenizer.tokenize("This is a sentence.")
        tokens = [Token("<S>")] + [t for t in tokens] + [Token("</S>")]
        indexer = SingleIdTokenIndexer(
            namespace="dep_labels", feature_name="dep_", default_value="NONE"
        )
        counter = defaultdict(lambda: defaultdict(int))
        for token in tokens:
            indexer.count_vocab_items(token, counter)

        assert counter["dep_labels"] == {
            "ROOT": 1,
            "nsubj": 1,
            "det": 1,
            "NONE": 2,
            "attr": 1,
            "punct": 1,
        }

    def test_tokens_to_indices_with_non_default_feature_name(self):
        tokenizer = SpacyTokenizer(parse=True)
        tokens = tokenizer.tokenize("This is a sentence.")
        tokens = [t for t in tokens] + [Token("</S>")]
        vocab = Vocabulary()
        root_index = vocab.add_token_to_namespace("ROOT", namespace="dep_labels")
        none_index = vocab.add_token_to_namespace("NONE", namespace="dep_labels")
        indexer = SingleIdTokenIndexer(
            namespace="dep_labels", feature_name="dep_", default_value="NONE"
        )
        assert indexer.tokens_to_indices([tokens[1]], vocab) == {"tokens": [root_index]}
        assert indexer.tokens_to_indices([tokens[-1]], vocab) == {"tokens": [none_index]}

    def test_crashes_with_empty_feature_value_and_no_default(self):
        tokenizer = SpacyTokenizer(parse=True)
        tokens = tokenizer.tokenize("This is a sentence.")
        tokens = [t for t in tokens] + [Token("</S>")]
        vocab = Vocabulary()
        vocab.add_token_to_namespace("ROOT", namespace="dep_labels")
        vocab.add_token_to_namespace("NONE", namespace="dep_labels")
        indexer = SingleIdTokenIndexer(namespace="dep_labels", feature_name="dep_")
        with pytest.raises(ValueError):
            indexer.tokens_to_indices([tokens[-1]], vocab)

    def test_no_namespace_means_no_counting(self):
        tokenizer = SpacyTokenizer(parse=True)
        tokens = tokenizer.tokenize("This is a sentence.")
        tokens = [Token("<S>")] + [t for t in tokens] + [Token("</S>")]
        indexer = SingleIdTokenIndexer(namespace=None, feature_name="text_id")

        def fail():
            assert False

        counter = defaultdict(fail)
        for token in tokens:
            indexer.count_vocab_items(token, counter)

    def test_no_namespace_means_no_indexing(self):
        indexer = SingleIdTokenIndexer(namespace=None, feature_name="text_id")
        assert indexer.tokens_to_indices([Token(text_id=23)], None) == {"tokens": [23]}
