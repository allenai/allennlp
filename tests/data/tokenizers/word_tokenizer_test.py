# pylint: disable=no-self-use,invalid-name

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import WordTokenizer

class TestWordTokenizer(AllenNlpTestCase):
    def test_passes_through_correctly(self):
        tokenizer = WordTokenizer(start_tokens=['@@', '%%'], end_tokens=['^^'])
        sentence = "this (sentence) has 'crazy' \"punctuation\"."
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        expected_tokens = ["@@", "%%", "this", "(", "sentence", ")", "has", "'", "crazy", "'", "\"",
                           "punctuation", "\"", ".", "^^"]
        assert tokens == expected_tokens

    def test_stems_and_filters_correctly(self):
        tokenizer = WordTokenizer.from_params(Params({'word_stemmer': {'type': 'porter'},
                                                      'word_filter': {'type': 'stopwords'}}))
        sentence = "this (sentence) has 'crazy' \"punctuation\"."
        expected_tokens = ["sentenc", "ha", "crazi", "punctuat"]
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens
