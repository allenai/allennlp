# pylint: disable=no-self-use,invalid-name
import os
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

    def test_batch_tokenization(self):
        tokenizer = WordTokenizer()
        sentences = ["This is a sentence",
                     "This isn't a sentence.",
                     "This is the 3rd sentence."
                     "Here's the 'fourth' sentence."]
        batch_tokenized = tokenizer.batch_tokenize(sentences)
        separately_tokenized = [tokenizer.tokenize(sentence) for sentence in sentences]
        assert len(batch_tokenized) == len(separately_tokenized)
        for batch_sentence, separate_sentence in zip(batch_tokenized, separately_tokenized):
            assert len(batch_sentence) == len(separate_sentence)
            for batch_word, separate_word in zip(batch_sentence, separate_sentence):
                assert batch_word.text == separate_word.text

    def test_stems_and_filters_stopwords_correctly(self):
        tokenizer = WordTokenizer.from_params(Params({'word_stemmer': {'type': 'porter'},
                                                      'word_filter': {'type': 'stopwords'}}))
        sentence = "this (sentence) has 'crazy' \"punctuation\"."
        expected_tokens = ["sentenc", "ha", "crazi", "punctuat"]
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

    def test_stems_and_filters_stopwords_from_file_correctly(self):
        with open('/tmp/tmp_stopwords.txt', 'w+') as f:
            for word in ["has", "this"]:
                f.write(word + "\n")
        tokenizer = WordTokenizer.from_params(Params({'word_stemmer': {'type': 'porter'},
                                                      'word_filter': {'type': 'stopwords',
                                                                      'stopword_file': "/tmp/tmp_stopwords.txt"}}))
        sentence = "this (sentence) has 'crazy' \"punctuation\"."
        expected_tokens = ['(', 'sentenc', ')', "'", 'crazi', "'", '"', 'punctuat', '"', '.']
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        os.remove('/tmp/tmp_stopwords.txt')
        assert tokens == expected_tokens

    def test_filters_digits_correctly(self):
        tokenizer = WordTokenizer.from_params(Params({'word_filter': {'type': 'regex',
                                                                      'patterns': ["[\d]+"]}}))
        sentence = "this 45 (sentence) has 9943434 \"punctuations\"."
        expected_tokens = ["this", "(", "sentence", ")", "has", '"', "punctuations", '"', '.']
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

    def test_filters_punctuation_correctly(self):
        tokenizer = WordTokenizer.from_params(Params({'word_filter': {'type': 'regex',
                                                                      'patterns': ['\(|\)|\"|\.']}}))
        sentence = "this 45 (sentence) has 9943434 \"punctuations\"."
        expected_tokens = ["this", "45", "sentence",  "has", "9943434", "punctuations"]
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

    def test_filters_multiple_patterns_correctly(self):
        tokenizer = WordTokenizer.from_params(Params({'word_filter': {'type': 'regex',
                                                                      'patterns': ['\(|\)|\"|\.', "[\d+]"]}}))
        sentence = "this 45 (sentence) has 9943434 \"punctuations\"."
        expected_tokens = ["this", "sentence",  "has", "punctuations"]
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens
    
    def test_filters_no_match_correctly(self):
        tokenizer = WordTokenizer.from_params(Params({'word_filter': {'type': 'regex', 'patterns': ['&']}}))
        sentence = "this 45 (sentence) has 9943434 \"punctuations\"."
        expected_tokens = ["this", "45", "(", "sentence", ")", "has",  "9943434", '"', "punctuations", '"', '.']
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens
