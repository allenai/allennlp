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
