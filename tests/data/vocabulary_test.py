# pylint: disable=no-self-use,invalid-name
import codecs
import os
import gzip
from copy import deepcopy

import pytest
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Instance, Token
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import CharacterTokenizer
from allennlp.data.vocabulary import Vocabulary, _NamespaceDependentDefaultDict, DEFAULT_OOV_TOKEN
from allennlp.common.params import Params
from allennlp.common.checks import ConfigurationError


class TestVocabulary(AllenNlpTestCase):
    def setUp(self):
        token_indexer = SingleIdTokenIndexer("tokens")
        text_field = TextField([Token(t) for t in ["a", "a", "a", "a", "b", "b", "c", "c", "c"]],
                               {"tokens": token_indexer})
        self.instance = Instance({"text": text_field})
        self.dataset = Batch([self.instance])
        super(TestVocabulary, self).setUp()

    def test_from_dataset_respects_min_count(self):

        vocab = Vocabulary.from_instances(self.dataset, min_count={'tokens': 4})
        words = vocab.get_index_to_token_vocabulary().values()
        assert 'a' in words
        assert 'b' not in words
        assert 'c' not in words

        vocab = Vocabulary.from_instances(self.dataset, min_count=None)
        words = vocab.get_index_to_token_vocabulary().values()
        assert 'a' in words
        assert 'b' in words
        assert 'c' in words

    def test_from_dataset_respects_exclusive_embedding_file(self):
        embeddings_filename = self.TEST_DIR + "embeddings.gz"
        with gzip.open(embeddings_filename, 'wb') as embeddings_file:
            embeddings_file.write("a 1.0 2.3 -1.0\n".encode('utf-8'))
            embeddings_file.write("b 0.1 0.4 -4.0\n".encode('utf-8'))

        vocab = Vocabulary.from_instances(self.dataset,
                                          min_count={'tokens': 4},
                                          pretrained_files={'tokens': embeddings_filename},
                                          only_include_pretrained_words=True)
        words = vocab.get_index_to_token_vocabulary().values()
        assert 'a' in words
        assert 'b' not in words
        assert 'c' not in words

        vocab = Vocabulary.from_instances(self.dataset,
                                          pretrained_files={'tokens': embeddings_filename},
                                          only_include_pretrained_words=True)
        words = vocab.get_index_to_token_vocabulary().values()
        assert 'a' in words
        assert 'b' in words
        assert 'c' not in words

    def test_from_dataset_respects_inclusive_embedding_file(self):
        embeddings_filename = self.TEST_DIR + "embeddings.gz"
        with gzip.open(embeddings_filename, 'wb') as embeddings_file:
            embeddings_file.write("a 1.0 2.3 -1.0\n".encode('utf-8'))
            embeddings_file.write("b 0.1 0.4 -4.0\n".encode('utf-8'))

        vocab = Vocabulary.from_instances(self.dataset,
                                          min_count={'tokens': 4},
                                          pretrained_files={'tokens': embeddings_filename},
                                          only_include_pretrained_words=False)
        words = vocab.get_index_to_token_vocabulary().values()
        assert 'a' in words
        assert 'b' in words
        assert 'c' not in words

        vocab = Vocabulary.from_instances(self.dataset,
                                          pretrained_files={'tokens': embeddings_filename},
                                          only_include_pretrained_words=False)
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

    def test_set_from_file_reads_padded_files(self):
        # pylint: disable=protected-access
        vocab_filename = self.TEST_DIR + 'vocab_file'
        with codecs.open(vocab_filename, 'w', 'utf-8') as vocab_file:
            vocab_file.write('<S>\n')
            vocab_file.write('</S>\n')
            vocab_file.write('<UNK>\n')
            vocab_file.write('a\n')
            vocab_file.write('tricky\x0bchar\n')
            vocab_file.write('word\n')
            vocab_file.write('another\n')

        vocab = Vocabulary()
        vocab.set_from_file(vocab_filename, is_padded=True, oov_token="<UNK>")

        assert vocab._oov_token == DEFAULT_OOV_TOKEN
        assert vocab.get_token_index("random string") == 3
        assert vocab.get_token_index("<S>") == 1
        assert vocab.get_token_index("</S>") == 2
        assert vocab.get_token_index(DEFAULT_OOV_TOKEN) == 3
        assert vocab.get_token_index("a") == 4
        assert vocab.get_token_index("tricky\x0bchar") == 5
        assert vocab.get_token_index("word") == 6
        assert vocab.get_token_index("another") == 7
        assert vocab.get_token_from_index(0) == vocab._padding_token
        assert vocab.get_token_from_index(1) == "<S>"
        assert vocab.get_token_from_index(2) == "</S>"
        assert vocab.get_token_from_index(3) == DEFAULT_OOV_TOKEN
        assert vocab.get_token_from_index(4) == "a"
        assert vocab.get_token_from_index(5) == "tricky\x0bchar"
        assert vocab.get_token_from_index(6) == "word"
        assert vocab.get_token_from_index(7) == "another"

    def test_set_from_file_reads_non_padded_files(self):
        # pylint: disable=protected-access
        vocab_filename = self.TEST_DIR + 'vocab_file'
        with codecs.open(vocab_filename, 'w', 'utf-8') as vocab_file:
            vocab_file.write('B-PERS\n')
            vocab_file.write('I-PERS\n')
            vocab_file.write('O\n')
            vocab_file.write('B-ORG\n')
            vocab_file.write('I-ORG\n')

        vocab = Vocabulary()
        vocab.set_from_file(vocab_filename, is_padded=False, namespace='tags')
        assert vocab.get_token_index("B-PERS", namespace='tags') == 0
        assert vocab.get_token_index("I-PERS", namespace='tags') == 1
        assert vocab.get_token_index("O", namespace='tags') == 2
        assert vocab.get_token_index("B-ORG", namespace='tags') == 3
        assert vocab.get_token_index("I-ORG", namespace='tags') == 4
        assert vocab.get_token_from_index(0, namespace='tags') == "B-PERS"
        assert vocab.get_token_from_index(1, namespace='tags') == "I-PERS"
        assert vocab.get_token_from_index(2, namespace='tags') == "O"
        assert vocab.get_token_from_index(3, namespace='tags') == "B-ORG"
        assert vocab.get_token_from_index(4, namespace='tags') == "I-ORG"

    def test_saving_and_loading(self):
        # pylint: disable=protected-access
        vocab_dir = os.path.join(self.TEST_DIR, 'vocab_save')

        vocab = Vocabulary(non_padded_namespaces=["a", "c"])
        vocab.add_token_to_namespace("a0", namespace="a")  # non-padded, should start at 0
        vocab.add_token_to_namespace("a1", namespace="a")
        vocab.add_token_to_namespace("a2", namespace="a")
        vocab.add_token_to_namespace("b2", namespace="b")  # padded, should start at 2
        vocab.add_token_to_namespace("b3", namespace="b")

        vocab.save_to_files(vocab_dir)
        vocab2 = Vocabulary.from_files(vocab_dir)

        assert vocab2._non_padded_namespaces == ["a", "c"]

        # Check namespace a.
        assert vocab2.get_vocab_size(namespace='a') == 3
        assert vocab2.get_token_from_index(0, namespace='a') == 'a0'
        assert vocab2.get_token_from_index(1, namespace='a') == 'a1'
        assert vocab2.get_token_from_index(2, namespace='a') == 'a2'
        assert vocab2.get_token_index('a0', namespace='a') == 0
        assert vocab2.get_token_index('a1', namespace='a') == 1
        assert vocab2.get_token_index('a2', namespace='a') == 2

        # Check namespace b.
        assert vocab2.get_vocab_size(namespace='b') == 4  # (unk + padding + two tokens)
        assert vocab2.get_token_from_index(0, namespace='b') == vocab._padding_token
        assert vocab2.get_token_from_index(1, namespace='b') == vocab._oov_token
        assert vocab2.get_token_from_index(2, namespace='b') == 'b2'
        assert vocab2.get_token_from_index(3, namespace='b') == 'b3'
        assert vocab2.get_token_index(vocab._padding_token, namespace='b') == 0
        assert vocab2.get_token_index(vocab._oov_token, namespace='b') == 1
        assert vocab2.get_token_index('b2', namespace='b') == 2
        assert vocab2.get_token_index('b3', namespace='b') == 3

        # Check the dictionaries containing the reverse mapping are identical.
        assert vocab.get_index_to_token_vocabulary("a") == vocab2.get_index_to_token_vocabulary("a")
        assert vocab.get_index_to_token_vocabulary("b") == vocab2.get_index_to_token_vocabulary("b")

    def test_saving_and_loading_works_with_byte_encoding(self):
        # We're going to set a vocabulary from a TextField using byte encoding, index it, save the
        # vocab, load the vocab, then index the text field again, and make sure we get the same
        # result.
        tokenizer = CharacterTokenizer(byte_encoding='utf-8')
        token_indexer = TokenCharactersIndexer(character_tokenizer=tokenizer)
        tokens = [Token(t) for t in ["Øyvind", "für", "汉字"]]
        text_field = TextField(tokens, {"characters": token_indexer})
        dataset = Batch([Instance({"sentence": text_field})])
        vocab = Vocabulary.from_instances(dataset)
        text_field.index(vocab)
        indexed_tokens = deepcopy(text_field._indexed_tokens)  # pylint: disable=protected-access

        vocab_dir = os.path.join(self.TEST_DIR, 'vocab_save')
        vocab.save_to_files(vocab_dir)
        vocab2 = Vocabulary.from_files(vocab_dir)
        text_field2 = TextField(tokens, {"characters": token_indexer})
        text_field2.index(vocab2)
        indexed_tokens2 = deepcopy(text_field2._indexed_tokens)  # pylint: disable=protected-access
        assert indexed_tokens == indexed_tokens2

    def test_from_params(self):
        # Save a vocab to check we can load it from_params.
        vocab_dir = os.path.join(self.TEST_DIR, 'vocab_save')
        vocab = Vocabulary(non_padded_namespaces=["a", "c"])
        vocab.add_token_to_namespace("a0", namespace="a")  # non-padded, should start at 0
        vocab.add_token_to_namespace("a1", namespace="a")
        vocab.add_token_to_namespace("a2", namespace="a")
        vocab.add_token_to_namespace("b2", namespace="b")  # padded, should start at 2
        vocab.add_token_to_namespace("b3", namespace="b")
        vocab.save_to_files(vocab_dir)

        params = Params({"directory_path": vocab_dir})
        vocab2 = Vocabulary.from_params(params)
        assert vocab.get_index_to_token_vocabulary("a") == vocab2.get_index_to_token_vocabulary("a")
        assert vocab.get_index_to_token_vocabulary("b") == vocab2.get_index_to_token_vocabulary("b")

        # Test case where we build a vocab from a dataset.
        vocab2 = Vocabulary.from_params(Params({}), self.dataset)
        assert vocab2.get_index_to_token_vocabulary("tokens") == {0: '@@PADDING@@',
                                                                  1: '@@UNKNOWN@@',
                                                                  2: 'a', 3: 'c', 4: 'b'}
        # Test from_params raises when we have neither a dataset and a vocab_directory.
        with pytest.raises(ConfigurationError):
            _ = Vocabulary.from_params(Params({}))

        # Test from_params raises when there are any other dict keys
        # present apart from 'vocabulary_directory' and we aren't calling from_dataset.
        with pytest.raises(ConfigurationError):
            _ = Vocabulary.from_params(Params({"directory_path": vocab_dir, "min_count": {'tokens': 2}}))
