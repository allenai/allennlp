# pylint: disable=no-self-use,invalid-name
import gzip

import numpy
import pytest
import torch
from torch.autograd import Variable
import h5py

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary
from allennlp.modules.token_embedders.embedding import Embedding, _read_pretrained_embedding_file


class TestEmbedding(AllenNlpTestCase):
    # pylint: disable=protected-access
    def test_get_embedding_layer_uses_correct_embedding_dim(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace('word1')
        vocab.add_token_to_namespace('word2')
        embeddings_filename = self.TEST_DIR + "embeddings.gz"
        with gzip.open(embeddings_filename, 'wb') as embeddings_file:
            embeddings_file.write("word1 1.0 2.3 -1.0\n".encode('utf-8'))
            embeddings_file.write("word2 0.1 0.4 -4.0\n".encode('utf-8'))
        embedding_weights = _read_pretrained_embedding_file(embeddings_filename, 3, vocab)
        assert tuple(embedding_weights.size()) == (4, 3)  # 4 because of padding and OOV
        with pytest.raises(ConfigurationError):
            _read_pretrained_embedding_file(embeddings_filename, 4, vocab)

    def test_forward_works_with_projection_layer(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace('the')
        vocab.add_token_to_namespace('a')
        params = Params({
                'pretrained_file': 'tests/fixtures/glove.6B.300d.sample.txt.gz',
                'embedding_dim': 300,
                'projection_dim': 20
                })
        embedding_layer = Embedding.from_params(vocab, params)
        input_tensor = Variable(torch.LongTensor([[3, 2, 1, 0]]))
        embedded = embedding_layer(input_tensor).data.numpy()
        assert embedded.shape == (1, 4, 20)

        input_tensor = Variable(torch.LongTensor([[[3, 2, 1, 0]]]))
        embedded = embedding_layer(input_tensor).data.numpy()
        assert embedded.shape == (1, 1, 4, 20)

    def test_embedding_layer_actually_initializes_word_vectors_correctly(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("word")
        vocab.add_token_to_namespace("word2")
        embeddings_filename = self.TEST_DIR + "embeddings.gz"
        with gzip.open(embeddings_filename, 'wb') as embeddings_file:
            embeddings_file.write("word 1.0 2.3 -1.0\n".encode('utf-8'))
        params = Params({
                'pretrained_file': embeddings_filename,
                'embedding_dim': 3,
                })
        embedding_layer = Embedding.from_params(vocab, params)
        word_vector = embedding_layer.weight.data[vocab.get_token_index("word")]
        assert numpy.allclose(word_vector.numpy(), numpy.array([1.0, 2.3, -1.0]))
        word_vector = embedding_layer.weight.data[vocab.get_token_index("word2")]
        assert not numpy.allclose(word_vector.numpy(), numpy.array([1.0, 2.3, -1.0]))

    def test_get_embedding_layer_initializes_unseen_words_randomly_not_zero(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("word")
        vocab.add_token_to_namespace("word2")
        embeddings_filename = self.TEST_DIR + "embeddings.gz"
        with gzip.open(embeddings_filename, 'wb') as embeddings_file:
            embeddings_file.write("word 1.0 2.3 -1.0\n".encode('utf-8'))
        params = Params({
                'pretrained_file': embeddings_filename,
                'embedding_dim': 3,
                })
        embedding_layer = Embedding.from_params(vocab, params)
        word_vector = embedding_layer.weight.data[vocab.get_token_index("word2")]
        assert not numpy.allclose(word_vector.numpy(), numpy.array([0.0, 0.0, 0.0]))

    def test_read_hdf5_format_file(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("word")
        vocab.add_token_to_namespace("word2")
        embeddings_filename = self.TEST_DIR + "embeddings.hdf5"
        embeddings = numpy.random.rand(vocab.get_vocab_size(), 5)
        with h5py.File(embeddings_filename, 'w') as fout:
            _ = fout.create_dataset(
                    'embedding', embeddings.shape, dtype='float32', data=embeddings
            )

        params = Params({
                'pretrained_file': embeddings_filename,
                'embedding_dim': 5,
                })
        embedding_layer = Embedding.from_params(vocab, params)
        assert numpy.allclose(embedding_layer.weight.data.numpy(), embeddings)

    def test_read_hdf5_raises_on_invalid_shape(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("word")
        embeddings_filename = self.TEST_DIR + "embeddings.hdf5"
        embeddings = numpy.random.rand(vocab.get_vocab_size(), 10)
        with h5py.File(embeddings_filename, 'w') as fout:
            _ = fout.create_dataset(
                    'embedding', embeddings.shape, dtype='float32', data=embeddings
            )

        params = Params({
                'pretrained_file': embeddings_filename,
                'embedding_dim': 5,
                })
        with pytest.raises(ConfigurationError):
            _ = Embedding.from_params(vocab, params)
