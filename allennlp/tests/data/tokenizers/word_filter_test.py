import os
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers.word_filter import RegexFilter
from allennlp.data.tokenizers.word_filter import StopwordFilter
from allennlp.data.tokenizers.token import Token


class TestWordFilter(AllenNlpTestCase):
    def setUp(self):
        super(TestWordFilter, self).setUp()
        self.sentence = ["this", "45", "(", "sentence", ")", "has", "9943434", '"', "punctuations", '"', "."]
        self.sentence = list(map(Token, self.sentence))

    def test_filters_digits_correctly(self):
        filter_ = RegexFilter(patterns=["\d+"])
        expected_tokens = ["this", "(", "sentence", ")", "has", '"', "punctuations", '"', '.']
        tokens = [t.text for t in filter_.filter_words(self.sentence)]
        assert tokens == expected_tokens

    def test_filters_punctuation_correctly(self):
        filter_ = RegexFilter(patterns=['\(|\)|\"|\.'])
        expected_tokens = ["this", "45", "sentence",  "has", "9943434", "punctuations"]
        tokens = [t.text for t in filter_.filter_words(self.sentence)]
        assert tokens == expected_tokens

    def test_filters_multiple_patterns_correctly(self):
        filter_ = RegexFilter(patterns=['\(|\)|\"|\.', "[\d+]"])
        expected_tokens = ["this", "sentence",  "has", "punctuations"]
        tokens = [t.text for t in filter_.filter_words(self.sentence)]
        assert tokens == expected_tokens

    def test_filters_no_match_correctly(self):
        filter_ = RegexFilter(patterns=['&'])
        expected_tokens = ["this", "45", "(", "sentence", ")", "has",  "9943434", '"', "punctuations", '"', '.']
        tokens = [t.text for t in filter_.filter_words(self.sentence)]
        assert tokens == expected_tokens

    def test_filters_stopwords_from_file_correctly(self):
        with open('/tmp/tmp_stopwords.txt', 'w+') as f:
            for word in ["has", "this"]:
                f.write(word + "\n")
        filter_ = StopwordFilter(stopword_file="/tmp/tmp_stopwords.txt",
                                 tokens_to_add=["punctuations"])
        expected_tokens = ['45', '(', 'sentence', ')', '9943434', '"', '"', '.']
        tokens = [t.text for t in filter_.filter_words(self.sentence)]
        os.remove('/tmp/tmp_stopwords.txt')
        assert tokens == expected_tokens
