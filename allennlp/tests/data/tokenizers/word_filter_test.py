import os
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers.word_filter import RegexFilter
from allennlp.data.tokenizers.word_filter import StopwordFilter
from allennlp.data.tokenizers.token import Token


class TestWordFilter(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.sentence = [
            "this",
            "45",
            "(",
            "sentence",
            ")",
            "has",
            "9943434",
            '"',
            "punctuations",
            '"',
            ".",
        ]
        self.sentence = list(map(Token, self.sentence))

    def test_filters_digits_correctly(self):
        filter_ = RegexFilter(patterns=[r"\d+"])
        expected_tokens = ["this", "(", "sentence", ")", "has", '"', "punctuations", '"', "."]
        tokens = [t.text for t in filter_.filter_words(self.sentence)]
        assert tokens == expected_tokens

    def test_filters_punctuation_correctly(self):
        filter_ = RegexFilter(patterns=[r"\(|\)|\"|\."])
        expected_tokens = ["this", "45", "sentence", "has", "9943434", "punctuations"]
        tokens = [t.text for t in filter_.filter_words(self.sentence)]
        assert tokens == expected_tokens

    def test_filters_multiple_patterns_correctly(self):
        filter_ = RegexFilter(patterns=[r"\(|\)|\"|\.", r"[\d+]"])
        expected_tokens = ["this", "sentence", "has", "punctuations"]
        tokens = [t.text for t in filter_.filter_words(self.sentence)]
        assert tokens == expected_tokens

    def test_filters_no_match_correctly(self):
        filter_ = RegexFilter(patterns=[r"&"])
        expected_tokens = [
            "this",
            "45",
            "(",
            "sentence",
            ")",
            "has",
            "9943434",
            '"',
            "punctuations",
            '"',
            ".",
        ]
        tokens = [t.text for t in filter_.filter_words(self.sentence)]
        assert tokens == expected_tokens

    def test_filters_from_file_correctly(self):
        stopword_file = os.path.join(self.TEST_DIR, "stopwords.txt")
        with open(stopword_file, "w+") as file_:
            for word in ["has", "this", "I"]:
                file_.write(word + "\n")
        filter_ = StopwordFilter(
            stopword_file=stopword_file, tokens_to_add=["punctuations", "PUNCTUATIONS"]
        )
        assert filter_.stopwords == {"has", "this", "i", "punctuations"}
        expected_tokens = ["45", "(", "sentence", ")", "9943434", '"', '"', "."]
        tokens = [t.text for t in filter_.filter_words(self.sentence)]
        assert tokens == expected_tokens
