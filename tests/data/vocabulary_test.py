# pylint: disable=no-self-use,invalid-name
import codecs

from allennlp.data.dataset import Dataset
from allennlp.data.fields.text_field import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.vocabulary import _NamespaceDependentDefaultDict
from allennlp.testing.test_case import AllenNlpTestCase


class TestVocabulary(AllenNlpTestCase):

    def setUp(self):
        token_indexer = SingleIdTokenIndexer("tokens")
        text_field = TextField(["a", "a", "a", "a", "b", "b", "c", "c", "c"], {"tokens": token_indexer})
        self.instance = Instance({"text": text_field})
        self.dataset = Dataset([self.instance])
        super(TestVocabulary, self).setUp()

    def test_from_dataset_respects_min_count(self):

        vocab = Vocabulary.from_dataset(self.dataset, min_count=4)
        words = vocab.get_index_to_token_vocabulary().values()
        assert 'a' in words
        assert 'b' not in words
        assert 'c' not in words

        vocab = Vocabulary.from_dataset(self.dataset, min_count=1)
        words = vocab.get_index_to_token_vocabulary().values()
        assert 'a' in words
        assert 'b' in words
        assert 'c' in words

    def test_add_word_to_index_gives_consistent_results(self):
        vocab = Vocabulary()
        initial_vocab_size = vocab.get_vocab_size()
        word_index = vocab.add_token_to_namespace("word")
        assert "word" in vocab.get_index_to_token_vocabulary().values()
        assert vocab.get_token_index("word") == word_index
        assert vocab.get_token_from_index(word_index) == "word"
        assert vocab.get_vocab_size() == initial_vocab_size + 1

        # Now add it again, and make sure nothing changes.
        vocab.add_token_to_namespace("word")
        assert "word" in vocab.get_index_to_token_vocabulary().values()
        assert vocab.get_token_index("word") == word_index
        assert vocab.get_token_from_index(word_index) == "word"
        assert vocab.get_vocab_size() == initial_vocab_size + 1

    def test_namespaces(self):
        vocab = Vocabulary()
        initial_vocab_size = vocab.get_vocab_size()
        word_index = vocab.add_token_to_namespace("word", namespace='1')
        assert "word" in vocab.get_index_to_token_vocabulary(namespace='1').values()
        assert vocab.get_token_index("word", namespace='1') == word_index
        assert vocab.get_token_from_index(word_index, namespace='1') == "word"
        assert vocab.get_vocab_size(namespace='1') == initial_vocab_size + 1

        # Now add it again, in a different namespace and a different word, and make sure it's like
        # new.
        word2_index = vocab.add_token_to_namespace("word2", namespace='2')
        word_index = vocab.add_token_to_namespace("word", namespace='2')
        assert "word" in vocab.get_index_to_token_vocabulary(namespace='2').values()
        assert "word2" in vocab.get_index_to_token_vocabulary(namespace='2').values()
        assert vocab.get_token_index("word", namespace='2') == word_index
        assert vocab.get_token_index("word2", namespace='2') == word2_index
        assert vocab.get_token_from_index(word_index, namespace='2') == "word"
        assert vocab.get_token_from_index(word2_index, namespace='2') == "word2"
        assert vocab.get_vocab_size(namespace='2') == initial_vocab_size + 2

    def test_namespace_dependent_default_dict(self):
        default_dict = _NamespaceDependentDefaultDict(["bar", "*baz"], lambda: 7, lambda: 3)
        # 'foo' is not a padded namespace
        assert default_dict["foo"] == 7
        # "baz" is a direct match with a padded namespace
        assert default_dict["baz"] == 3
        # the following match the wildcard "*baz"
        assert default_dict["bar"] == 3
        assert default_dict["foobaz"] == 3

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
