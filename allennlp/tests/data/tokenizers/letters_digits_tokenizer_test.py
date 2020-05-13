from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers.letters_digits_tokenizer import LettersDigitsTokenizer
from allennlp.data.tokenizers.token import Token


class TestLettersDigitsTokenizer(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.word_tokenizer = LettersDigitsTokenizer()

    def test_tokenize_handles_complex_punctuation(self):
        sentence = "this (sentence) has 'crazy' \"punctuation\"."
        expected_tokens = [
            "this",
            "(",
            "sentence",
            ")",
            "has",
            "'",
            "crazy",
            "'",
            '"',
            "punctuation",
            '"',
            ".",
        ]
        tokens = [t.text for t in self.word_tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

    def test_tokenize_handles_unicode_letters(self):
        sentence = "HAL9000   and    Ångström"
        expected_tokens = [
            Token("HAL", 0),
            Token("9000", 3),
            Token("and", 10),
            Token("Ångström", 17),
        ]
        tokens = self.word_tokenizer.tokenize(sentence)
        assert [t.text for t in tokens] == [t.text for t in expected_tokens]
        assert [t.idx for t in tokens] == [t.idx for t in expected_tokens]

    def test_tokenize_handles_splits_all_punctuation(self):
        sentence = "wouldn't.[have] -3.45(m^2)"
        expected_tokens = [
            "wouldn",
            "'",
            "t",
            ".",
            "[",
            "have",
            "]",
            "-",
            "3",
            ".",
            "45",
            "(",
            "m",
            "^",
            "2",
            ")",
        ]
        tokens = [t.text for t in self.word_tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens
