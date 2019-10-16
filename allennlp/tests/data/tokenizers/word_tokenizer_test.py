import spacy

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.letters_digits_tokenizer import LettersDigitsTokenizer
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.data.tokenizers.pretrained_transformer_pre_tokenizer import OpenAIPreTokenizer
from allennlp.data.tokenizers.pretrained_transformer_pre_tokenizer import BertPreTokenizer


class TestLettersDigitsTokenizer(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
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


class TestSpacyTokenizer(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.word_tokenizer = SpacyTokenizer()

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
        tokens = self.word_tokenizer.tokenize(sentence)
        token_text = [t.text for t in tokens]
        assert token_text == expected_tokens
        for token in tokens:
            start = token.idx
            end = start + len(token.text)
            assert sentence[start:end] == token.text

    def test_tokenize_handles_contraction(self):
        # note that "would've" is kept together, while "ain't" is not.
        sentence = "it ain't joe's problem; would been yesterday"
        expected_tokens = [
            "it",
            "ai",
            "n't",
            "joe",
            "'s",
            "problem",
            ";",
            "would",
            "been",
            "yesterday",
        ]
        tokens = [t.text for t in self.word_tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

    def test_tokenize_handles_multiple_contraction(self):
        sentence = "wouldn't've"
        expected_tokens = ["would", "n't", "'ve"]
        tokens = [t.text for t in self.word_tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

    def test_tokenize_handles_final_apostrophe(self):
        sentence = "the jones' house"
        expected_tokens = ["the", "jones", "'", "house"]
        tokens = [t.text for t in self.word_tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

    def test_tokenize_removes_whitespace_tokens(self):
        sentence = "the\n jones'   house  \x0b  55"
        expected_tokens = ["the", "jones", "'", "house", "55"]
        tokens = [t.text for t in self.word_tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

    def test_tokenize_handles_special_cases(self):
        # note that the etc. doesn't quite work --- we can special case this if we want.
        sentence = "Mr. and Mrs. Jones, etc., went to, e.g., the store"
        expected_tokens = [
            "Mr.",
            "and",
            "Mrs.",
            "Jones",
            ",",
            "etc",
            ".",
            ",",
            "went",
            "to",
            ",",
            "e.g.",
            ",",
            "the",
            "store",
        ]
        tokens = [t.text for t in self.word_tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

    def test_batch_tokenization(self):
        sentences = [
            "This is     a sentence",
            "This isn't a sentence.",
            "This is the 3rd     sentence." "Here's the 'fourth' sentence.",
        ]
        batch_split = self.word_tokenizer.batch_tokenize(sentences)
        separately_split = [self.word_tokenizer.tokenize(sentence) for sentence in sentences]
        assert len(batch_split) == len(separately_split)
        for batch_sentence, separate_sentence in zip(batch_split, separately_split):
            assert len(batch_sentence) == len(separate_sentence)
            for batch_word, separate_word in zip(batch_sentence, separate_sentence):
                assert batch_word.text == separate_word.text

    def test_keep_spacy_tokens(self):
        word_tokenizer = SpacyTokenizer()
        sentence = "This should be an allennlp Token"
        tokens = word_tokenizer.tokenize(sentence)
        assert tokens
        assert all(isinstance(token, Token) for token in tokens)

        word_tokenizer = SpacyTokenizer(keep_spacy_tokens=True)
        sentence = "This should be a spacy Token"
        tokens = word_tokenizer.tokenize(sentence)
        assert tokens
        assert all(isinstance(token, spacy.tokens.Token) for token in tokens)


class TestOpenAiSpacyTokenizer(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.word_tokenizer = OpenAIPreTokenizer()

    def test_tokenize_handles_complex_punctuation(self):
        sentence = "This sentence ?a!?!"
        expected_tokens = ["This", "sentence", "?", "a", "!", "?", "!"]
        tokens = [t.text for t in self.word_tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens


class TestBertPreTokenizer(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.word_tokenizer = BertPreTokenizer()

    def test_never_split(self):
        sentence = "[unused0] [UNK] [SEP] [PAD] [CLS] [MASK]"
        expected_tokens = ["[", "unused0", "]", "[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
        tokens = [token.text for token in self.word_tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

    def test_do_lower_case(self):
        # BertPreTokenizer makes every token not in `never_split` to lowercase by default
        word_tokenizer = BertPreTokenizer(never_split=["[UNUSED0]"])
        sentence = "[UNUSED0] [UNK] [unused0]"
        expected_tokens = ["[UNUSED0]", "[", "unk", "]", "[", "unused0", "]"]
        tokens = [token.text for token in word_tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens
