from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import PretrainedTransformerTokenizer


class TestPretrainedTransformerTokenizer(AllenNlpTestCase):
    def test_splits_roberta(self):
        tokenizer = PretrainedTransformerTokenizer("roberta-base")

        sentence = "A, <mask> AllenNLP sentence."
        expected_tokens = [
            "<s>",
            "A",
            ",",
            " ",
            "<mask>",
            " Allen",
            "N",
            "LP",
            " sentence",
            ".",
            "</s>",
        ]
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

        # sentence pair
        sentence_1 = "A, <mask> AllenNLP sentence."
        sentence_2 = "A sentence."
        expected_tokens = [
            "<s>",
            "A",
            ",",
            " ",
            "<mask>",
            " Allen",
            "N",
            "LP",
            " sentence",
            ".",
            "</s>",
            "</s>",
            "A",
            " sentence",
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
            "NL",
            "P",
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
            "NL",
            "P",
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
            "A",
            ",",
            "[MASK]",
            "Allen",
            "NL",
            "P",
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
            "A",
            ",",
            "naïve",
            "[MASK]",
            "Allen",
            "NL",
            "P",
            "sentence",
            ".",
            "[SEP]",
        ]
        expected_idxs = [
            None,
            0,
            1,
            None,  # TODO
            9,
            16,
            21,
            23,
            25,
            33,
            None,
        ]
        tokenizer = PretrainedTransformerTokenizer("bert-base-uncased")
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
            "ï",
            "ve",
            "[MASK]",
            "Allen",
            "NL",
            "P",
            "sentence",
            ".",
            "[SEP]",
        ]
        expected_idxs = [None, 0, 1, 3, 5, 6, 9, 16, 21, 23, 25, 33, None]
        tokenizer = PretrainedTransformerTokenizer("bert-base-cased")
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
            " naïve",
            "<mask>",
            " Allen",
            "N",
            "LP",
            " sentence",
            ".",
            "</s>",
        ]
        expected_idxs = [None, 0, 1, None, 9, 16, 21, 22, 25, 33, None]
        tokenizer = PretrainedTransformerTokenizer("roberta-base")
        tokenized = tokenizer.tokenize(sentence)
        tokens = [t.text for t in tokenized]
        assert tokens == expected_tokens
        idxs = [t.idx for t in tokenized]
        assert idxs == expected_idxs

    def test_token_idx_wikipedia(self):
        sentence = (
            "Tokyo (東京 Tōkyō, English: /ˈtoʊkioʊ/,[7] Japanese: [toːkʲoː]), officially "
            "Tokyo Metropolis (東京都 Tōkyō-to), is one of the 47 prefectures of Japan."
        )
        for tokenizer_name in ["roberta-base", "bert-base-uncased", "bert-base-cased"]:
            tokenizer = PretrainedTransformerTokenizer(tokenizer_name)
            tokenized = tokenizer.tokenize(sentence)
            assert tokenized[-2].text == "."
            assert tokenized[-2].idx == len(sentence) - 1

    def test_token_idx_sentence_pairs(self):
        first_sentence = "I went to the zoo yesterday, but they had only one animal."
        second_sentence = "It was a shitzu."
        expected_tokens = [
            "<s>",
            "I",
            " went",
            " to",
            " the",
            " zoo",
            " yesterday",
            ",",
            " but",
            " they",
            " had",
            " only",
            " one",
            " animal",
            ".",
            "</s>",
            "</s>",
            "It",
            " was",
            " a",
            " sh",
            "itz",
            "u",
            ".",
            "</s>",
        ]
        expected_idxs = [
            None,
            0,
            2,
            7,
            10,
            14,
            18,
            27,
            29,
            33,
            38,
            42,
            47,
            51,
            57,
            None,
            None,
            58,
            61,
            65,
            67,
            69,
            72,
            73,
            None,
        ]

        tokenizer = PretrainedTransformerTokenizer("roberta-base")
        tokenized = tokenizer.tokenize_sentence_pair(first_sentence, second_sentence)
        tokens = [t.text for t in tokenized]
        assert tokens == expected_tokens
        idxs = [t.idx for t in tokenized]
        assert idxs == expected_idxs

        # Assert that the first and the second sentence are run together with no space in between.
        first_sentence_end_index = tokens.index("</s>") - 1
        second_sentence_start_index = first_sentence_end_index + 3
        assert (
            idxs[first_sentence_end_index] + len(tokens[first_sentence_end_index])
            == idxs[second_sentence_start_index]
        )

    def test_intra_word_tokenize(self):
        tokenizer = PretrainedTransformerTokenizer("bert-base-cased")

        sentence = "A, [MASK] AllenNLP sentence.".split(" ")
        expected_tokens = [
            "[CLS]",
            "A",
            ",",
            "[MASK]",
            "Allen",
            "NL",
            "P",
            "sentence",
            ".",
            "[SEP]",
        ]
        expected_offsets = [(1, 2), (3, 3), (4, 6), (7, 8)]
        tokens, offsets = tokenizer.intra_word_tokenize(sentence)
        tokens = [t.text for t in tokens]
        assert tokens == expected_tokens
        assert offsets == expected_offsets

        # sentence pair
        sentence_1 = "A, [MASK] AllenNLP sentence.".split(" ")
        sentence_2 = "A sentence.".split(" ")
        expected_tokens = [
            "[CLS]",
            "A",
            ",",
            "[MASK]",
            "Allen",
            "NL",
            "P",
            "sentence",
            ".",
            "[SEP]",
            "A",
            "sentence",
            ".",
            "[SEP]",
        ]
        expected_offsets_a = [(1, 2), (3, 3), (4, 6), (7, 8)]
        expected_offsets_b = [(10, 10), (11, 12)]
        tokens, offsets_a, offsets_b = tokenizer.intra_word_tokenize_sentence_pair(
            sentence_1, sentence_2
        )
        tokens = [t.text for t in tokens]
        assert tokens == expected_tokens
        assert offsets_a == expected_offsets_a
        assert offsets_b == expected_offsets_b

    def test_intra_word_tokenize_whitespaces(self):
        tokenizer = PretrainedTransformerTokenizer("bert-base-cased")

        sentence = ["A,", " ", "[MASK]", "AllenNLP", "\u007f", "sentence."]
        expected_tokens = [
            "[CLS]",
            "A",
            ",",
            "[UNK]",
            "[MASK]",
            "Allen",
            "NL",
            "P",
            "[UNK]",
            "sentence",
            ".",
            "[SEP]",
        ]
        expected_offsets = [(1, 2), (3, 3), (4, 4), (5, 7), (8, 8), (9, 10)]
        tokens, offsets = tokenizer.intra_word_tokenize(sentence)
        tokens = [t.text for t in tokens]
        assert tokens == expected_tokens
        assert offsets == expected_offsets

    def test_determine_num_special_tokens_added(self):
        tokenizer = PretrainedTransformerTokenizer("bert-base-cased")
        assert tokenizer._determine_num_special_tokens_added() == (1, 1, 1)
        tokenizer = PretrainedTransformerTokenizer("xlnet-base-cased")
        assert tokenizer._determine_num_special_tokens_added() == (0, 1, 2)

    def test_tokenizer_kwargs_default(self):
        text = "Hello there! General Kenobi."
        tokenizer = PretrainedTransformerTokenizer("bert-base-cased")
        original_tokens = [
            "[CLS]",
            "Hello",
            "there",
            "!",
            "General",
            "Ken",
            "ob",
            "i",
            ".",
            "[SEP]",
        ]
        tokenized = [token.text for token in tokenizer.tokenize(text)]
        assert tokenized == original_tokens

    def test_from_params_kwargs(self):
        PretrainedTransformerTokenizer.from_params(
            Params({"model_name": "bert-base-uncased", "tokenizer_kwargs": {"max_len": 10}})
        )
