# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import ModelTestCase
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
from allennlp.data.tokenizers import WordTokenizer
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

        assert indexed_tokens["bert"] == [2, 3, 5, 6, 8, 9, 2, 15, 10, 11, 14, 1]
        assert indexed_tokens["bert-offsets"] == [0, 1, 2, 3, 4, 5, 6, 9, 10, 11]

        token_indexer = PretrainedBertIndexer(str(vocab_path), use_starting_offsets=True)

        indexed_tokens = token_indexer.tokens_to_indices(tokens, vocab, "bert")

        assert indexed_tokens["bert"] == [2, 3, 5, 6, 8, 9, 2, 15, 10, 11, 14, 1]
        assert indexed_tokens["bert-offsets"] == [0, 1, 2, 3, 4, 5, 6, 7, 10, 11]
