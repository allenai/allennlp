from allennlp.common.testing import ModelTestCase
from allennlp.data.token_indexers.wordpiece_indexer import (
    PretrainedBertIndexer,
    _get_token_type_ids,
)
from allennlp.data.tokenizers import WordTokenizer, Token
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
from allennlp.data.vocabulary import Vocabulary


class TestBertIndexer(ModelTestCase):
    def test_starting_ending_offsets(self):
        tokenizer = WordTokenizer(word_splitter=BertBasicWordSplitter())

        #           2   3     5     6   8      9    2  15 10 11 14   1
        sentence = "the quick brown fox jumped over the laziest lazy elmo"
        tokens = tokenizer.tokenize(sentence)

        vocab = Vocabulary()
        vocab_path = self.FIXTURES_ROOT / "bert" / "vocab.txt"
        token_indexer = PretrainedBertIndexer(str(vocab_path))

        indexed_tokens = token_indexer.tokens_to_indices(tokens, vocab, "bert")

        # 16 = [CLS], 17 = [SEP]
        assert indexed_tokens["bert"] == [16, 2, 3, 5, 6, 8, 9, 2, 15, 10, 11, 14, 1, 17]
        assert indexed_tokens["bert-offsets"] == [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]

        token_indexer = PretrainedBertIndexer(str(vocab_path), use_starting_offsets=True)

        indexed_tokens = token_indexer.tokens_to_indices(tokens, vocab, "bert")

        assert indexed_tokens["bert"] == [16, 2, 3, 5, 6, 8, 9, 2, 15, 10, 11, 14, 1, 17]
        assert indexed_tokens["bert-offsets"] == [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]

    def test_eq(self):
        vocab_path = self.FIXTURES_ROOT / "bert" / "vocab.txt"
        indexer1 = PretrainedBertIndexer(str(vocab_path))
        indexer2 = PretrainedBertIndexer(str(vocab_path))
        assert indexer1 == indexer2

    def test_do_lowercase(self):
        # Our default tokenizer doesn't handle lowercasing.
        tokenizer = WordTokenizer()

        # Quick is UNK because of capitalization
        #           2   1     5     6   8      9    2  15 10 11 14   1
        sentence = "the Quick brown fox jumped over the laziest lazy elmo"
        tokens = tokenizer.tokenize(sentence)

        vocab = Vocabulary()
        vocab_path = self.FIXTURES_ROOT / "bert" / "vocab.txt"
        token_indexer = PretrainedBertIndexer(str(vocab_path), do_lowercase=False)

        indexed_tokens = token_indexer.tokens_to_indices(tokens, vocab, "bert")

        # Quick should get 1 == OOV
        assert indexed_tokens["bert"] == [16, 2, 1, 5, 6, 8, 9, 2, 15, 10, 11, 14, 1, 17]

        # Does lowercasing by default
        token_indexer = PretrainedBertIndexer(str(vocab_path))
        indexed_tokens = token_indexer.tokens_to_indices(tokens, vocab, "bert")

        # Now Quick should get indexed correctly as 3 ( == "quick")
        assert indexed_tokens["bert"] == [16, 2, 3, 5, 6, 8, 9, 2, 15, 10, 11, 14, 1, 17]

    def test_never_lowercase(self):
        # Our default tokenizer doesn't handle lowercasing.
        tokenizer = WordTokenizer()

        #            2 15 10 11  6
        sentence = "the laziest fox"

        tokens = tokenizer.tokenize(sentence)
        tokens.append(Token("[PAD]"))  # have to do this b/c tokenizer splits it in three

        vocab = Vocabulary()
        vocab_path = self.FIXTURES_ROOT / "bert" / "vocab.txt"
        token_indexer = PretrainedBertIndexer(str(vocab_path), do_lowercase=True)

        indexed_tokens = token_indexer.tokens_to_indices(tokens, vocab, "bert")

        # PAD should get recognized and not lowercased      # [PAD]
        assert indexed_tokens["bert"] == [16, 2, 15, 10, 11, 6, 0, 17]

        # Unless we manually override the never lowercases
        token_indexer = PretrainedBertIndexer(
            str(vocab_path), do_lowercase=True, never_lowercase=()
        )
        indexed_tokens = token_indexer.tokens_to_indices(tokens, vocab, "bert")

        # now PAD should get lowercased and be UNK          # [UNK]
        assert indexed_tokens["bert"] == [16, 2, 15, 10, 11, 6, 1, 17]

    def test_get_token_type_ids(self):
        wordpiece_ids = [0, 1, 2, 3, 4, 5]

        # when the `separator` is at the end
        separator_ids = [5]
        desired_token_type_ids = [0, 0, 0, 0, 0, 0]
        assert _get_token_type_ids(wordpiece_ids, separator_ids) == desired_token_type_ids

        # when the `separator` does not appear
        separator_ids = [6]
        desired_token_type_ids = [0, 0, 0, 0, 0, 0]
        assert _get_token_type_ids(wordpiece_ids, separator_ids) == desired_token_type_ids

        # when the `separator` contains multi-tokens and at the end
        separator_ids = [3, 4, 5]
        desired_token_type_ids = [0, 0, 0, 0, 0, 0]
        assert _get_token_type_ids(wordpiece_ids, separator_ids) == desired_token_type_ids

        # when the `separator` contains multi-tokens and does not appear
        separator_ids = [6, 7, 8]
        desired_token_type_ids = [0, 0, 0, 0, 0, 0]
        assert _get_token_type_ids(wordpiece_ids, separator_ids) == desired_token_type_ids

        wordpiece_ids = [0, 1, 2, 3, 1, 2, 3, 1, 2]

        # when the `separator` appears many times
        separator_ids = [2]
        desired_token_type_ids = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        assert _get_token_type_ids(wordpiece_ids, separator_ids) == desired_token_type_ids

        # when the `separator` contains multi-tokens and appears many times
        separator_ids = [1, 2]
        desired_token_type_ids = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        assert _get_token_type_ids(wordpiece_ids, separator_ids) == desired_token_type_ids

        # edge case: the `separator` contains multi-tokens and
        # some of them are same as the end of the word pieces
        separator_ids = [2, 3]
        desired_token_type_ids = [0, 0, 0, 0, 1, 1, 1, 2, 2]
        assert _get_token_type_ids(wordpiece_ids, separator_ids) == desired_token_type_ids

    def test_token_type_ids(self):
        tokenizer = WordTokenizer()

        sentence = "the laziest  fox"

        tokens = tokenizer.tokenize(sentence)
        #           2   15 10 11  6   17    2   15 10 11  6
        #           the laziest   fox [SEP] the laziest   fox
        tokens = (
            tokens + [Token("[SEP]")] + tokens
        )  # have to do this b/c tokenizer splits `[SEP]` in three

        vocab = Vocabulary()
        vocab_path = self.FIXTURES_ROOT / "bert" / "vocab.txt"
        token_indexer = PretrainedBertIndexer(str(vocab_path))

        indexed_tokens = token_indexer.tokens_to_indices(tokens, vocab, "bert")

        #                                          [CLS] 2, 15, 10, 11, 6, 17, 2  15, 10, 11, 6, [SEP]
        assert indexed_tokens["bert-type-ids"] == [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    def test_sliding_window(self):
        tokenizer = WordTokenizer(word_splitter=BertBasicWordSplitter())

        sentence = "the quickest quick brown [SEP] jumped over the lazy dog"
        tokens = tokenizer.tokenize(sentence)

        vocab = Vocabulary()
        vocab_path = self.FIXTURES_ROOT / "bert" / "vocab.txt"
        token_indexer = PretrainedBertIndexer(
            str(vocab_path),
            truncate_long_sequences=False,
            use_starting_offsets=False,
            max_pieces=10,
        )

        indexed_tokens = token_indexer.tokens_to_indices(tokens, vocab, "bert")

        # 16 = [CLS], 17 = [SEP]
        # 1 full window + 1 half window with start/end tokens
        # [CLS] the quick est quick brown [SEP] jumped over [SEP]
        assert indexed_tokens["bert"] == [
            16,
            2,
            3,
            4,
            3,
            5,
            17,
            8,
            9,
            17,
            # [CLS] brown [SEP] jumped over the lazy dog [SEP]
            16,
            5,
            17,
            8,
            9,
            2,
            14,
            12,
            17,
        ]
        assert indexed_tokens["bert-offsets"] == [1, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        # The extra [SEP]s shouldn't pollute the token-type-ids
        # [CLS] the quick est quick brown [SEP] jumped over [SEP]
        assert indexed_tokens["bert-type-ids"] == [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            # [CLS] brown [SEP] jumped over the lazy dog [SEP]
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
        ]

    def test_truncate_window(self):
        tokenizer = WordTokenizer(word_splitter=BertBasicWordSplitter())

        sentence = "the quickest quick brown fox jumped over the lazy dog"
        tokens = tokenizer.tokenize(sentence)

        vocab = Vocabulary()
        vocab_path = self.FIXTURES_ROOT / "bert" / "vocab.txt"
        token_indexer = PretrainedBertIndexer(
            str(vocab_path), truncate_long_sequences=True, use_starting_offsets=True, max_pieces=10
        )

        indexed_tokens = token_indexer.tokens_to_indices(tokens, vocab, "bert")

        # 16 = [CLS], 17 = [SEP]
        # 1 full window + 1 half window with start/end tokens
        assert indexed_tokens["bert"] == [16, 2, 3, 4, 3, 5, 6, 8, 9, 17]
        assert indexed_tokens["bert-offsets"] == [1, 2, 4, 5, 6, 7, 8]
        assert indexed_tokens["bert-type-ids"] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        token_indexer = PretrainedBertIndexer(
            str(vocab_path), truncate_long_sequences=True, use_starting_offsets=False, max_pieces=10
        )

        indexed_tokens = token_indexer.tokens_to_indices(tokens, vocab, "bert")

        # 16 = [CLS], 17 = [SEP]
        # 1 full window + 1 half window with start/end tokens
        assert indexed_tokens["bert"] == [16, 2, 3, 4, 3, 5, 6, 8, 9, 17]
        assert indexed_tokens["bert-offsets"] == [1, 3, 4, 5, 6, 7, 8]

    def test_truncate_window_dont_split_wordpieces(self):
        """
        Tests if the sentence is not truncated inside of the word with 2 or
        more wordpieces.
        """

        tokenizer = WordTokenizer(word_splitter=BertBasicWordSplitter())

        sentence = "the quickest quick brown fox jumped over the quickest dog"
        tokens = tokenizer.tokenize(sentence)

        vocab = Vocabulary()
        vocab_path = self.FIXTURES_ROOT / "bert" / "vocab.txt"
        token_indexer = PretrainedBertIndexer(
            str(vocab_path), truncate_long_sequences=True, use_starting_offsets=True, max_pieces=12
        )

        indexed_tokens = token_indexer.tokens_to_indices(tokens, vocab, "bert")

        # 16 = [CLS], 17 = [SEP]
        # 1 full window + 1 half window with start/end tokens
        assert indexed_tokens["bert"] == [16, 2, 3, 4, 3, 5, 6, 8, 9, 2, 17]
        # We could fit one more piece here, but we don't, not to have a cut
        # in the middle of the word
        assert indexed_tokens["bert-offsets"] == [1, 2, 4, 5, 6, 7, 8, 9]
        assert indexed_tokens["bert-type-ids"] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        token_indexer = PretrainedBertIndexer(
            str(vocab_path), truncate_long_sequences=True, use_starting_offsets=False, max_pieces=12
        )

        indexed_tokens = token_indexer.tokens_to_indices(tokens, vocab, "bert")

        # 16 = [CLS], 17 = [SEP]
        # 1 full window + 1 half window with start/end tokens
        assert indexed_tokens["bert"] == [16, 2, 3, 4, 3, 5, 6, 8, 9, 2, 17]
        # We could fit one more piece here, but we don't, not to have a cut
        # in the middle of the word
        assert indexed_tokens["bert-offsets"] == [1, 3, 4, 5, 6, 7, 8, 9]

    def test_truncate_window_fit_two_wordpieces(self):
        """
        Tests if the both `use_starting_offsets` options work properly when last
        word in the truncated sentence consists of two wordpieces.
        """

        tokenizer = WordTokenizer(word_splitter=BertBasicWordSplitter())

        sentence = "the quickest quick brown fox jumped over the quickest dog"
        tokens = tokenizer.tokenize(sentence)

        vocab = Vocabulary()
        vocab_path = self.FIXTURES_ROOT / "bert" / "vocab.txt"
        token_indexer = PretrainedBertIndexer(
            str(vocab_path), truncate_long_sequences=True, use_starting_offsets=True, max_pieces=13
        )

        indexed_tokens = token_indexer.tokens_to_indices(tokens, vocab, "bert")

        # 16 = [CLS], 17 = [SEP]
        # 1 full window + 1 half window with start/end tokens
        assert indexed_tokens["bert"] == [16, 2, 3, 4, 3, 5, 6, 8, 9, 2, 3, 4, 17]
        assert indexed_tokens["bert-offsets"] == [1, 2, 4, 5, 6, 7, 8, 9, 10]
        assert indexed_tokens["bert-type-ids"] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        token_indexer = PretrainedBertIndexer(
            str(vocab_path), truncate_long_sequences=True, use_starting_offsets=False, max_pieces=13
        )

        indexed_tokens = token_indexer.tokens_to_indices(tokens, vocab, "bert")

        # 16 = [CLS], 17 = [SEP]
        # 1 full window + 1 half window with start/end tokens
        assert indexed_tokens["bert"] == [16, 2, 3, 4, 3, 5, 6, 8, 9, 2, 3, 4, 17]
        assert indexed_tokens["bert-offsets"] == [1, 3, 4, 5, 6, 7, 8, 9, 11]

    def test_indexes_empty_sequence(self):
        vocab = Vocabulary()
        vocab_path = self.FIXTURES_ROOT / "bert" / "vocab.txt"
        token_indexer = PretrainedBertIndexer(str(vocab_path))

        indexed_tokens = token_indexer.tokens_to_indices([], vocab, "bert")
        assert indexed_tokens == {
            "bert": [16, 17],  # [CLS], [SEP]
            "bert-offsets": [],  # no tokens => no offsets
            "bert-type-ids": [0, 0],  # just 0s for start and end
            "mask": [],  # no tokens => no mask
        }
