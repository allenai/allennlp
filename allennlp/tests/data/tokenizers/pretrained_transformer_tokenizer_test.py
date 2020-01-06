import gc

import time

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
            "ĠnaÃ¯ve",  # RoBERTa has a funny way of encoding combining characters.
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

    def test_token_idx_wikipedia(self):
        # This will produce lots of problems with the index calculation. We check whether it catches back up at the
        # end.
        sentence = "Tokyo (東京 Tōkyō, English: /ˈtoʊkioʊ/,[7] Japanese: [toːkʲoː]), officially Tokyo Metropolis (東京都 Tōkyō-to), is one of the 47 prefectures of Japan."
        for tokenizer_name in ["roberta-base", "bert-base-uncased", "bert-base-cased"]:
            tokenizer = PretrainedTransformerTokenizer(
                tokenizer_name, calculate_character_offsets=True
            )
            tokenized = tokenizer.tokenize(sentence)
            assert tokenized[-2].text == "."
            assert tokenized[-2].idx == len(sentence) - 1

    def test_token_idx_performance(self):
        text = """
            Tokyo (東京 Tōkyō, English: /ˈtoʊkioʊ/,[7] Japanese: [toːkʲoː]), officially Tokyo Metropolis (東京都
            Tōkyō-to), is one of the 47 prefectures of Japan. It has served as the Japanese capital since
            1869,[8][9] its urban area housing the Emperor of Japan and the Japanese government. Tokyo forms part
            of the Kantō region on the southeastern side of Japan's main island, Honshu, and includes the Izu
            Islands and Ogasawara Islands.[10] Tokyo was formerly named Edo when Shōgun Tokugawa Ieyasu made the
            city his headquarters in 1603. It became the capital after Emperor Meiji moved his seat to the city
            from Kyoto in 1868; at that time Edo was renamed Tokyo.[11] The Tokyo Metropolis formed in 1943 from
            the merger of the former Tokyo Prefecture (東京府 Tōkyō-fu) and the city of Tokyo (東京市 Tōkyō-shi).
            Tokyo is often referred to as a city but is officially known and governed as a "metropolitan
            prefecture", which differs from and combines elements of a city and a prefecture, a characteristic
            unique to Tokyo."""

        tokenizer_with_idx = PretrainedTransformerTokenizer(
            "roberta-base", calculate_character_offsets=True
        )
        tokenizer_without_idx = PretrainedTransformerTokenizer(
            "roberta-base", calculate_character_offsets=False
        )

        gc.collect()
        start = time.monotonic()
        for i in range(200):
            tokenizer_without_idx.tokenize(text)
        without_idx_time = time.monotonic() - start

        gc.collect()
        start = time.monotonic()
        for i in range(200):
            tokenizer_with_idx.tokenize(text)
        with_idx_time = time.monotonic() - start

        assert with_idx_time <= 2 * without_idx_time

    def test_token_idx_sentence_pairs(self):
        first_sentence = "I went to the zoo yesterday, but they had only one animal."
        second_sentence = "It was a shitzu."
        expected_tokens = [
            "<s>",
            "I",
            "Ġwent",
            "Ġto",
            "Ġthe",
            "Ġzoo",
            "Ġyesterday",
            ",",
            "Ġbut",
            "Ġthey",
            "Ġhad",
            "Ġonly",
            "Ġone",
            "Ġanimal",
            ".",
            "</s>",
            "</s>",
            "It",
            "Ġwas",
            "Ġa",
            "Ġsh",
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

        tokenizer = PretrainedTransformerTokenizer("roberta-base", calculate_character_offsets=True)
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
