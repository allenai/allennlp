# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import ModelTestCase
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
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
        vocab_path = self.FIXTURES_ROOT / 'bert' / 'vocab.txt'
        token_indexer = PretrainedBertIndexer(str(vocab_path))

        indexed_tokens = token_indexer.tokens_to_indices(tokens, vocab, "bert")

        # 16 = [CLS], 17 = [SEP]
        assert indexed_tokens["bert"] == [16, 2, 3, 5, 6, 8, 9, 2, 15, 10, 11, 14, 1, 17]
        assert indexed_tokens["bert-offsets"] == [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]

        token_indexer = PretrainedBertIndexer(str(vocab_path), use_starting_offsets=True)

        indexed_tokens = token_indexer.tokens_to_indices(tokens, vocab, "bert")

        assert indexed_tokens["bert"] == [16, 2, 3, 5, 6, 8, 9, 2, 15, 10, 11, 14, 1, 17]
        assert indexed_tokens["bert-offsets"] == [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]


    def test_do_lowercase(self):
        # Our default tokenizer doesn't handle lowercasing.
        tokenizer = WordTokenizer()

        # Quick is UNK because of capitalization
        #           2   1     5     6   8      9    2  15 10 11 14   1
        sentence = "the Quick brown fox jumped over the laziest lazy elmo"
        tokens = tokenizer.tokenize(sentence)

        vocab = Vocabulary()
        vocab_path = self.FIXTURES_ROOT / 'bert' / 'vocab.txt'
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
        vocab_path = self.FIXTURES_ROOT / 'bert' / 'vocab.txt'
        token_indexer = PretrainedBertIndexer(str(vocab_path), do_lowercase=True)

        indexed_tokens = token_indexer.tokens_to_indices(tokens, vocab, "bert")

        # PAD should get recognized and not lowercased      # [PAD]
        assert indexed_tokens["bert"] == [16, 2, 15, 10, 11, 6, 0, 17]

        # Unless we manually override the never lowercases
        token_indexer = PretrainedBertIndexer(str(vocab_path), do_lowercase=True, never_lowercase=())
        indexed_tokens = token_indexer.tokens_to_indices(tokens, vocab, "bert")

        # now PAD should get lowercased and be UNK          # [UNK]
        assert indexed_tokens["bert"] == [16, 2, 15, 10, 11, 6, 1, 17]
