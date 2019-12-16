from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import HuggingfaceTransformersTokenizer


class TestHuggingTransformersTokenizer(AllenNlpTestCase):
    def test_splits_roberta(self):
        tokenizer = HuggingfaceTransformersTokenizer("roberta-base")

        sentence = "A, <mask> AllenNLP sentence."
        expected_tokens = ["A", ",", "<mask>", "Allen", "N", "LP", "Ġsentence", "."]
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

        # tokenize with offsets
        tokens, offsets = list(
            map(list, zip(*[(t.text, t.idx) for t in tokenizer.tokenize_with_offsets(sentence)]))
        )
        assert tokens == expected_tokens
        assert offsets == [0, 1, 3, 10, 15, 16, 19, 27]

    def test_splits_cased_bert(self):
        tokenizer = HuggingfaceTransformersTokenizer("bert-base-cased")

        sentence = "A, [MASK] AllenNLP sentence."
        expected_tokens = ["A", ",", "[MASK]", "Allen", "##NL", "##P", "sentence", "."]
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

        # tokenize with offsets
        tokens, offsets = list(
            map(list, zip(*[(t.text, t.idx) for t in tokenizer.tokenize_with_offsets(sentence)]))
        )
        assert tokens == expected_tokens
        assert offsets == [0, 1, 3, 10, 15, 17, 19, 27]

    def test_splits_uncased_bert(self):
        sentence = "A, [MASK] AllenNLP sentence."
        expected_tokens = [
            "a",
            ",",
            "[MASK]",
            "allen",
            "##nl",
            "##p",
            "sentence",
            ".",
        ]
        tokenizer = HuggingfaceTransformersTokenizer("bert-base-uncased")
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

        # tokenize with offsets
        tokens, offsets = list(
            map(list, zip(*[(t.text, t.idx) for t in tokenizer.tokenize_with_offsets(sentence)]))
        )
        assert tokens == expected_tokens
        assert offsets == [0, 1, 3, 10, 15, 17, 19, 27]

    def test_tokenize_with_offsets_prefix_issue(self):
        """ For words that have a wordpiece tokenization that
            doesn't contain the tokenization of its prefixes.
            Example for XLNet:
            text = "1603"
            tokens = ["▁16", "03"]
            tokenization for "160": ["▁160"]
        """
        sentence = "1603"
        expected_tokens = ["▁16", "03"]
        tokenizer = HuggingfaceTransformersTokenizer("xlnet-base-cased")

        # tokenize with offsets
        tokens, offsets = list(
            map(list, zip(*[(t.text, t.idx) for t in tokenizer.tokenize_with_offsets(sentence)]))
        )
        assert tokens == expected_tokens
        assert offsets == [0, 2]

    def test_tokenize_with_offsets_additional_char_issue(self):
        """ For cases in which the current token won't be produced
            without an additional character that is only part of the
            text that corresponds to the next tokens.
            Example for XLNet:
            text = "How many points did the buccaneers need to tie in the first?"
            tokens = [..., '▁the', '▁', 'bu', 'cca', 'ne', 'ers', ...]
            target_tokens = ['▁']
            comparison_tokens = ['▁', 'b']
            prev_comparison_tokens = ['']
        """
        sentence = "How many points did the buccaneers need to tie in the first?"
        expected_tokens = [
            "▁How",
            "▁many",
            "▁points",
            "▁did",
            "▁the",
            "▁",
            "bu",
            "cca",
            "ne",
            "ers",
            "▁need",
            "▁to",
            "▁tie",
            "▁in",
            "▁the",
            "▁first",
            "?",
        ]
        tokenizer = HuggingfaceTransformersTokenizer("xlnet-base-cased")

        # tokenize with offsets
        tokens, offsets = list(
            map(list, zip(*[(t.text, t.idx) for t in tokenizer.tokenize_with_offsets(sentence)]))
        )
        assert tokens == expected_tokens
        assert offsets == [0, 4, 9, 16, 20, 24, 24, 26, 29, 31, 35, 40, 43, 47, 50, 54, 59]
