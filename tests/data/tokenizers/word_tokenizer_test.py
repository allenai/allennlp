# pylint: disable=no-self-use,invalid-name

from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.common.params import Params

class TestWordProcessor:
    def test_passes_through_correctly(self):
        word_processor = WordTokenizer()
        sentence = "this (sentence) has 'crazy' \"punctuation\"."
        tokens = word_processor.tokenize(sentence)
        expected_tokens = ["this", "(", "sentence", ")", "has", "'", "crazy", "'", "\"",
                           "punctuation", "\"", "."]
        assert tokens == expected_tokens

    def test_stems_and_filters_correctly(self):
        word_processor = WordTokenizer.from_params(Params({'word_stemmer': {'type': 'porter'},
                                                           'word_filter': {'type': 'stopwords'}}))
        sentence = "this (sentence) has 'crazy' \"punctuation\"."
        expected_tokens = ["sentenc", "ha", "crazi", "punctuat"]
        tokens = word_processor.tokenize(sentence)
        assert tokens == expected_tokens
