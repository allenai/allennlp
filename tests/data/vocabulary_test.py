# pylint: disable=no-self-use,invalid-name
import codecs

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields.text_field import TextField
from allennlp.data.instance import Instance
from allennlp.data.dataset import Dataset
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.testing.test_case import DeepQaTestCase


class TestVocabulary(DeepQaTestCase):

    def setUp(self):
        token_indexer = SingleIdTokenIndexer("tokens")
        text_field = TextField(["a", "a", "a", "a", "b", "b", "c", "c", "c"], [token_indexer])
        self.instance = Instance({"text": text_field})
        self.dataset = Dataset([self.instance])
        super(TestVocabulary, self).setUp()

    def test_from_dataset_respects_min_count(self):

        vocab = Vocabulary.from_dataset(self.dataset, min_count=4)
        assert 'a' in vocab.tokens_in_namespace()
        assert 'b' not in vocab.tokens_in_namespace()
        assert 'c' not in vocab.tokens_in_namespace()

        vocab = Vocabulary.from_dataset(self.dataset, min_count=1)
        assert 'a' in vocab.tokens_in_namespace()
        assert 'b' in vocab.tokens_in_namespace()
        assert 'c' in vocab.tokens_in_namespace()

    def test_add_word_to_index_gives_consistent_results(self):
        vocab = Vocabulary()
        initial_vocab_size = vocab.get_vocab_size()
        word_index = vocab.add_token_to_namespace("word")
        assert "word" in vocab.tokens_in_namespace()
        assert vocab.get_token_index("word") == word_index
        assert vocab.get_token_from_index(word_index) == "word"
        assert vocab.get_vocab_size() == initial_vocab_size + 1

        # Now add it again, and make sure nothing changes.
        vocab.add_token_to_namespace("word")
        assert "word" in vocab.tokens_in_namespace()
        assert vocab.get_token_index("word") == word_index
        assert vocab.get_token_from_index(word_index) == "word"
        assert vocab.get_vocab_size() == initial_vocab_size + 1

    def test_namespaces(self):
        vocab = Vocabulary()
        initial_vocab_size = vocab.get_vocab_size()
        word_index = vocab.add_token_to_namespace("word", namespace='1')
        assert "word" in vocab.tokens_in_namespace(namespace='1')
        assert vocab.get_token_index("word", namespace='1') == word_index
        assert vocab.get_token_from_index(word_index, namespace='1') == "word"
        assert vocab.get_vocab_size(namespace='1') == initial_vocab_size + 1

        # Now add it again, in a different namespace and a different word, and make sure it's like
        # new.
        word2_index = vocab.add_token_to_namespace("word2", namespace='2')
        word_index = vocab.add_token_to_namespace("word", namespace='2')
        assert "word" in vocab.tokens_in_namespace(namespace='2')
        assert "word2" in vocab.tokens_in_namespace(namespace='2')
        assert vocab.get_token_index("word", namespace='2') == word_index
        assert vocab.get_token_index("word2", namespace='2') == word2_index
        assert vocab.get_token_from_index(word_index, namespace='2') == "word"
        assert vocab.get_token_from_index(word2_index, namespace='2') == "word2"
        assert vocab.get_vocab_size(namespace='2') == initial_vocab_size + 2

    def test_unknown_token(self):
        # pylint: disable=protected-access
        # We're putting this behavior in a test so that the behavior is documented.  There is
        # solver code that depends in a small way on how we treat the unknown token, so any
        # breaking change to this behavior should break a test, so you know you've done something
        # that needs more consideration.
        vocab = Vocabulary()
        oov_token = vocab._oov_token
        oov_index = vocab.get_token_index(oov_token)
        assert oov_index == 1
        assert vocab.get_token_index("unseen word") == oov_index

    def test_set_from_file(self):
        # pylint: disable=protected-access
        vocab_filename = self.TEST_DIR + 'vocab_file'
        with codecs.open(vocab_filename, 'w', 'utf-8') as vocab_file:
            vocab_file.write('<S>\n')
            vocab_file.write('</S>\n')
            vocab_file.write('<UNK>\n')
            vocab_file.write('a\n')
            vocab_file.write('word\n')
            vocab_file.write('another\n')
        vocab = Vocabulary()
        vocab.set_from_file(vocab_filename, oov_token="<UNK>")
        assert vocab._oov_token == "<UNK>"
        assert vocab.get_token_index("random string") == 3
        assert vocab.get_token_index("<S>") == 1
        assert vocab.get_token_index("</S>") == 2
        assert vocab.get_token_index("<UNK>") == 3
        assert vocab.get_token_index("a") == 4
        assert vocab.get_token_index("word") == 5
        assert vocab.get_token_index("another") == 6
        assert vocab.get_token_from_index(0) == vocab._padding_token
        assert vocab.get_token_from_index(1) == "<S>"
        assert vocab.get_token_from_index(2) == "</S>"
        assert vocab.get_token_from_index(3) == "<UNK>"
        assert vocab.get_token_from_index(4) == "a"
        assert vocab.get_token_from_index(5) == "word"
        assert vocab.get_token_from_index(6) == "another"
