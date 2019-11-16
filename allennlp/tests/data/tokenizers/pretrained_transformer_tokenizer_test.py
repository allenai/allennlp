from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import PretrainedTransformerTokenizer


class TestPretrainedTransformerTokenizer(AllenNlpTestCase):
    def test_splits_roberta(self):
        tokenizer = PretrainedTransformerTokenizer("roberta-base")

        sentence = "A, <mask> AllenNLP sentence."
        expected_tokens = ["<s>", "A", ",", "<mask>", "Allen", "N", "LP", "Ġsentence", ".", "</s>"]
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

        # sentence pair
        sentence_1 = "A, <mask> AllenNLP sentence."
        sentence_2 = "A sentence."
        expected_tokens = [
            "<s>",
            "A",
            ",",
            "<mask>",
            "Allen",
            "N",
            "LP",
            "Ġsentence",
            ".",
            "</s>",
            "</s>",
            "A",
            "Ġsentence",
            ".",
            "</s>",
        ]
        tokens = [t.text for t in tokenizer.tokenize_sentence_pair(sentence_1, sentence_2)]
        assert tokens == expected_tokens

    def test_splits_cased_bert(self):
        tokenizer = PretrainedTransformerTokenizer("bert-base-cased")

        sentence = "A, [MASK] AllenNLP sentence."
        expected_tokens = [
            "[CLS]",
            "A",
            ",",
            "[MASK]",
            "Allen",
            "##NL",
            "##P",
            "sentence",
            ".",
            "[SEP]",
        ]
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

        # sentence pair
        sentence_1 = "A, [MASK] AllenNLP sentence."
        sentence_2 = "A sentence."
        expected_tokens = [
            "[CLS]",
            "A",
            ",",
            "[MASK]",
            "Allen",
            "##NL",
            "##P",
            "sentence",
            ".",
            "[SEP]",
            "A",
            "sentence",
            ".",
            "[SEP]",
        ]
        tokens = [t.text for t in tokenizer.tokenize_sentence_pair(sentence_1, sentence_2)]
        assert tokens == expected_tokens

    def test_splits_uncased_bert(self):
        sentence = "A, [MASK] AllenNLP sentence."
        expected_tokens = [
            "[CLS]",
            "a",
            ",",
            "[MASK]",
            "allen",
            "##nl",
            "##p",
            "sentence",
            ".",
            "[SEP]",
        ]
        tokenizer = PretrainedTransformerTokenizer("bert-base-uncased")
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens
