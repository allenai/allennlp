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

    def test_token_idx_bert_uncased(self):
        sentence = "A, naïve [MASK] AllenNLP sentence."
        expected_tokens = [
            "[CLS]",
            "a",
            ",",
            "naive",  # It normalizes the accent.
            "[MASK]",
            "allen",
            "##nl",
            "##p",
            "sentence",
            ".",
            "[SEP]",
        ]
        expected_idxs = [
            None,
            0,
            1,
            None,  # It can't find this one because of the normalized accent.
            9,
            16,
            21,
            23,
            25,
            33,
            None,
        ]
        tokenizer = PretrainedTransformerTokenizer(
            "bert-base-uncased", calculate_character_offsets=True
        )
        tokenized = tokenizer.tokenize(sentence)
        tokens = [t.text for t in tokenized]
        assert tokens == expected_tokens
        idxs = [t.idx for t in tokenized]
        assert idxs == expected_idxs

    def test_token_idx_bert_cased(self):
        sentence = "A, naïve [MASK] AllenNLP sentence."
        expected_tokens = [
            "[CLS]",
            "A",
            ",",
            "na",
            "##ï",  # Does not normalize the accent
            "##ve",
            "[MASK]",
            "Allen",
            "##NL",
            "##P",
            "sentence",
            ".",
            "[SEP]",
        ]
        expected_idxs = [None, 0, 1, 3, 5, 6, 9, 16, 21, 23, 25, 33, None]
        tokenizer = PretrainedTransformerTokenizer(
            "bert-base-cased", calculate_character_offsets=True
        )
        tokenized = tokenizer.tokenize(sentence)
        tokens = [t.text for t in tokenized]
        assert tokens == expected_tokens
        idxs = [t.idx for t in tokenized]
        assert idxs == expected_idxs

    def test_token_idx_roberta(self):
        sentence = "A, naïve <mask> AllenNLP sentence."
        expected_tokens = [
            "<s>",
            "A",
            ",",
            "ĠnaÃ¯ve",  # What happened here?
            "<mask>",
            "Allen",
            "N",
            "LP",
            "Ġsentence",
            ".",
            "</s>",
        ]
        expected_idxs = [None, 0, 1, None, 9, 16, 21, 22, 25, 33, None]
        tokenizer = PretrainedTransformerTokenizer("roberta-base", calculate_character_offsets=True)
        tokenized = tokenizer.tokenize(sentence)
        tokens = [t.text for t in tokenized]
        assert tokens == expected_tokens
        idxs = [t.idx for t in tokenized]
        assert idxs == expected_idxs
