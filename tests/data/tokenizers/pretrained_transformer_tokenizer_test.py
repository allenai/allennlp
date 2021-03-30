from typing import Iterable, List

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token
from allennlp.data.tokenizers import PretrainedTransformerTokenizer


class TestPretrainedTransformerTokenizer(AllenNlpTestCase):
    def test_splits_roberta(self):
        tokenizer = PretrainedTransformerTokenizer("roberta-base")

        sentence = "A, <mask> AllenNLP sentence."
        expected_tokens = [
            "<s>",
            "A",
            ",",
            "<mask>",
            "ĠAllen",
            "N",
            "LP",
            "Ġsentence",
            ".",
            "</s>",
        ]
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
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

    def test_splits_reformer_small(self):
        sentence = "A, [MASK] AllenNLP sentence."
        expected_tokens = [
            "▁A",
            ",",
            "▁",
            "<unk>",
            "M",
            "A",
            "S",
            "K",
            "<unk>",
            "▁A",
            "ll",
            "en",
            "N",
            "L",
            "P",
            "▁s",
            "ent",
            "en",
            "ce",
            ".",
        ]
        tokenizer = PretrainedTransformerTokenizer("google/reformer-crime-and-punishment")
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

    def test_token_idx_bert_uncased(self):
        sentence = "A, naïve [MASK] AllenNLP sentence."
        expected_tokens = [
            "[CLS]",
            "a",
            ",",
            "naive",  # BERT normalizes this away
            "[MASK]",
            "allen",
            "##nl",
            "##p",
            "sentence",
            ".",
            "[SEP]",
        ]
        expected_idxs = [None, 0, 1, 3, 9, 16, 21, 23, 25, 33, None]
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
            "##ï",
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
        tokenizer = PretrainedTransformerTokenizer("bert-base-cased")
        tokenized = tokenizer.tokenize(sentence)
        tokens = [t.text for t in tokenized]
        assert tokens == expected_tokens
        idxs = [t.idx for t in tokenized]
        assert idxs == expected_idxs

    def test_max_length(self):
        tokenizer = PretrainedTransformerTokenizer(
            "bert-base-cased", max_length=10, add_special_tokens=False
        )
        tokens = tokenizer.tokenize(
            "hi there, this should be at least 10 tokens, but some will be truncated"
        )
        assert len(tokens) == 10

    def test_no_max_length(self):
        tokenizer = PretrainedTransformerTokenizer(
            "bert-base-cased", max_length=None, add_special_tokens=False
        )
        # Even though the bert model has a max input length of 512, when we tokenize
        # with `max_length = None`, we should not get any truncation.
        tokens = tokenizer.tokenize(" ".join(["a"] * 550))
        assert len(tokens) == 550

    def test_token_idx_roberta(self):
        sentence = "A, naïve <mask> AllenNLP sentence."
        expected_tokens = [
            "<s>",
            "A",
            ",",
            "ĠnaÃ¯ve",  # RoBERTa mangles this. Or maybe it "encodes"?
            "<mask>",
            "ĠAllen",
            "N",
            "LP",
            "Ġsentence",
            ".",
            "</s>",
        ]
        expected_idxs = [None, 0, 1, 3, 9, 16, 21, 22, 25, 33, None]
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

    def test_intra_word_tokenize(self):
        tokenizer = PretrainedTransformerTokenizer("bert-base-cased")

        sentence = "A, [MASK] AllenNLP sentence.".split(" ")
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
            "##NL",
            "##P",
            "sentence",
            ".",
            "[SEP]",
            "A",  # 10
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
            "[MASK]",
            "Allen",
            "##NL",
            "##P",
            "sentence",
            ".",
            "[SEP]",
        ]
        expected_offsets = [(1, 2), None, (3, 3), (4, 6), None, (7, 8)]
        tokens, offsets = tokenizer.intra_word_tokenize(sentence)
        tokens = [t.text for t in tokens]
        assert tokens == expected_tokens
        assert offsets == expected_offsets

    def test_special_tokens_added(self):
        def get_token_ids(tokens: Iterable[Token]) -> List[int]:
            return [t.text_id for t in tokens]

        def get_type_ids(tokens: Iterable[Token]) -> List[int]:
            return [t.type_id for t in tokens]

        tokenizer = PretrainedTransformerTokenizer("bert-base-cased")
        assert get_token_ids(tokenizer.sequence_pair_start_tokens) == [101]
        assert get_token_ids(tokenizer.sequence_pair_mid_tokens) == [102]
        assert get_token_ids(tokenizer.sequence_pair_end_tokens) == [102]
        assert get_token_ids(tokenizer.single_sequence_start_tokens) == [101]
        assert get_token_ids(tokenizer.single_sequence_end_tokens) == [102]

        assert get_type_ids(tokenizer.sequence_pair_start_tokens) == [0]
        assert tokenizer.sequence_pair_first_token_type_id == 0
        assert get_type_ids(tokenizer.sequence_pair_mid_tokens) == [0]
        assert tokenizer.sequence_pair_second_token_type_id == 1
        assert get_type_ids(tokenizer.sequence_pair_end_tokens) == [1]

        assert get_type_ids(tokenizer.single_sequence_start_tokens) == [0]
        assert tokenizer.single_sequence_token_type_id == 0
        assert get_type_ids(tokenizer.single_sequence_end_tokens) == [0]

        tokenizer = PretrainedTransformerTokenizer("xlnet-base-cased")
        assert get_token_ids(tokenizer.sequence_pair_start_tokens) == []
        assert get_token_ids(tokenizer.sequence_pair_mid_tokens) == [4]
        assert get_token_ids(tokenizer.sequence_pair_end_tokens) == [4, 3]
        assert get_token_ids(tokenizer.single_sequence_start_tokens) == []
        assert get_token_ids(tokenizer.single_sequence_end_tokens) == [4, 3]

        assert get_type_ids(tokenizer.sequence_pair_start_tokens) == []
        assert tokenizer.sequence_pair_first_token_type_id == 0
        assert get_type_ids(tokenizer.sequence_pair_mid_tokens) == [0]
        assert tokenizer.sequence_pair_second_token_type_id == 1
        assert get_type_ids(tokenizer.sequence_pair_end_tokens) == [1, 2]

        assert get_type_ids(tokenizer.single_sequence_start_tokens) == []
        assert tokenizer.single_sequence_token_type_id == 0
        assert get_type_ids(tokenizer.single_sequence_end_tokens) == [0, 2]

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
            "##ob",
            "##i",
            ".",
            "[SEP]",
        ]
        tokenized = [token.text for token in tokenizer.tokenize(text)]
        assert tokenized == original_tokens

    def test_from_params_kwargs(self):
        PretrainedTransformerTokenizer.from_params(
            Params({"model_name": "bert-base-uncased", "tokenizer_kwargs": {"max_len": 10}})
        )
