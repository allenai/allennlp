# pylint: disable=no-self-use,invalid-name
import gzip

import numpy
import pytest

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders.embedding import get_pretrained_embedding_layer
from allennlp.testing.test_case import AllenNlpTestCase


class TestPretrainedEmbeddings(AllenNlpTestCase):
    # pylint: disable=protected-access
    def test_get_embedding_layer_uses_correct_embedding_dim(self):
        vocab = Vocabulary()
        embeddings_filename = self.TEST_DIR + "embeddings.gz"
        with gzip.open(embeddings_filename, 'wb') as embeddings_file:
            embeddings_file.write("word1 1.0 2.3 -1.0\n".encode('utf-8'))
            embeddings_file.write("word2 0.1 0.4 -4.0\n".encode('utf-8'))
        embedding_layer = get_pretrained_embedding_layer(embeddings_filename, vocab)
        assert embedding_layer.embedding_dim == 3

        with gzip.open(embeddings_filename, 'wb') as embeddings_file:
            embeddings_file.write("word1 1.0 2.3 -1.0 3.1\n".encode('utf-8'))
            embeddings_file.write("word2 0.1 0.4 -4.0 -1.2\n".encode('utf-8'))
        embedding_layer = get_pretrained_embedding_layer(embeddings_filename, vocab)
        assert embedding_layer.embedding_dim == 4

    def test_get_embedding_layer_crashes_when_embedding_dim_is_one(self):
        vocab = Vocabulary()
        embeddings_filename = self.TEST_DIR + "embeddings.gz"
        with gzip.open(embeddings_filename, 'wb') as embeddings_file:
            embeddings_file.write("dimensionality 3\n".encode('utf-8'))
            embeddings_file.write("word1 1.0 2.3 -1.0\n".encode('utf-8'))
            embeddings_file.write("word2 0.1 0.4 -4.0\n".encode('utf-8'))
        with pytest.raises(Exception):
            get_pretrained_embedding_layer(embeddings_filename, vocab)

    def test_get_embedding_layer_skips_inconsistent_lines(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("word1")
        vocab.add_token_to_namespace("word2")
        embeddings_filename = self.TEST_DIR + "embeddings.gz"
        with gzip.open(embeddings_filename, 'wb') as embeddings_file:
            embeddings_file.write("word1 1.0 2.3 -1.0\n".encode('utf-8'))
            embeddings_file.write("word2 0.1 0.4 \n".encode('utf-8'))
        embedding_layer = get_pretrained_embedding_layer(embeddings_filename, vocab)
        word_vector = embedding_layer.weight.data[vocab.get_token_index("word2")]
        assert not numpy.allclose(word_vector.numpy()[:2], numpy.array([0.1, 0.4]))

    def test_get_embedding_layer_actually_initializes_word_vectors_correctly(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("word")
        embeddings_filename = self.TEST_DIR + "embeddings.gz"
        with gzip.open(embeddings_filename, 'wb') as embeddings_file:
            embeddings_file.write("word 1.0 2.3 -1.0\n".encode('utf-8'))
        embedding_layer = get_pretrained_embedding_layer(embeddings_filename, vocab)
        word_vector = embedding_layer.weight.data[vocab.get_token_index("word")]
        assert numpy.allclose(word_vector.numpy(), numpy.array([1.0, 2.3, -1.0]))

    def test_get_embedding_layer_initializes_unseen_words_randomly_not_zero(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("word2")
        embeddings_filename = self.TEST_DIR + "embeddings.gz"
        with gzip.open(embeddings_filename, 'wb') as embeddings_file:
            embeddings_file.write("word 1.0 2.3 -1.0\n".encode('utf-8'))
        embedding_layer = get_pretrained_embedding_layer(embeddings_filename, vocab)
        word_vector = embedding_layer.weight.data[vocab.get_token_index("word2")]
        assert not numpy.allclose(word_vector.numpy(), numpy.array([0.0, 0.0, 0.0]))
