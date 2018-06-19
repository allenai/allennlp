# pylint: disable=no-self-use,invalid-name
import gzip
import warnings

import numpy
import pytest
import torch
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary
from allennlp.modules.token_embedders.embedding import (Embedding,
                                                        _read_pretrained_embeddings_file,
                                                        EmbeddingsTextFile,
                                                        format_embeddings_file_uri,
                                                        parse_embeddings_file_uri)


class TestEmbedding(AllenNlpTestCase):
    # pylint: disable=protected-access
    def test_get_embedding_layer_uses_correct_embedding_dim(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace('word1')
        vocab.add_token_to_namespace('word2')
        embeddings_filename = str(self.TEST_DIR / "embeddings.gz")
        with gzip.open(embeddings_filename, 'wb') as embeddings_file:
            embeddings_file.write("word1 1.0 2.3 -1.0\n".encode('utf-8'))
            embeddings_file.write("word2 0.1 0.4 -4.0\n".encode('utf-8'))
        embedding_weights = _read_pretrained_embeddings_file(embeddings_filename, 3, vocab)
        assert tuple(embedding_weights.size()) == (4, 3)  # 4 because of padding and OOV
        with pytest.raises(ConfigurationError):
            _read_pretrained_embeddings_file(embeddings_filename, 4, vocab)

    def test_forward_works_with_projection_layer(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace('the')
        vocab.add_token_to_namespace('a')
        params = Params({
                'pretrained_file': str(self.FIXTURES_ROOT / 'embeddings/glove.6B.300d.sample.txt.gz'),
                'embedding_dim': 300,
                'projection_dim': 20
                })
        embedding_layer = Embedding.from_params(vocab, params)
        input_tensor = torch.LongTensor([[3, 2, 1, 0]])
        embedded = embedding_layer(input_tensor).data.numpy()
        assert embedded.shape == (1, 4, 20)

        input_tensor = torch.LongTensor([[[3, 2, 1, 0]]])
        embedded = embedding_layer(input_tensor).data.numpy()
        assert embedded.shape == (1, 1, 4, 20)

    def test_embedding_layer_actually_initializes_word_vectors_correctly(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("word")
        vocab.add_token_to_namespace("word2")
        unicode_space = "\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0"
        vocab.add_token_to_namespace(unicode_space)
        embeddings_filename = str(self.TEST_DIR / "embeddings.gz")
        with gzip.open(embeddings_filename, 'wb') as embeddings_file:
            embeddings_file.write("word 1.0 2.3 -1.0\n".encode('utf-8'))
            embeddings_file.write(f"{unicode_space} 3.4 3.3 5.0\n".encode('utf-8'))
        params = Params({
                'pretrained_file': embeddings_filename,
                'embedding_dim': 3,
                })
        embedding_layer = Embedding.from_params(vocab, params)
        word_vector = embedding_layer.weight.data[vocab.get_token_index("word")]
        assert numpy.allclose(word_vector.numpy(), numpy.array([1.0, 2.3, -1.0]))
        word_vector = embedding_layer.weight.data[vocab.get_token_index(unicode_space)]
        assert numpy.allclose(word_vector.numpy(), numpy.array([3.4, 3.3, 5.0]))
        word_vector = embedding_layer.weight.data[vocab.get_token_index("word2")]
        assert not numpy.allclose(word_vector.numpy(), numpy.array([1.0, 2.3, -1.0]))

    def test_get_embedding_layer_initializes_unseen_words_randomly_not_zero(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace("word")
        vocab.add_token_to_namespace("word2")
        embeddings_filename = str(self.TEST_DIR / "embeddings.gz")
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
        embeddings_filename = str(self.TEST_DIR / "embeddings.hdf5")
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
        embeddings_filename = str(self.TEST_DIR / "embeddings.hdf5")
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

    def test_read_embedding_file_inside_archive(self):
        token2vec = {
                "think": torch.Tensor([0.143, 0.189, 0.555, 0.361, 0.472]),
                "make": torch.Tensor([0.878, 0.651, 0.044, 0.264, 0.872]),
                "difference": torch.Tensor([0.053, 0.162, 0.671, 0.110, 0.259]),
                "àèìòù": torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
                }
        vocab = Vocabulary()
        for token in token2vec:
            vocab.add_token_to_namespace(token)

        params = Params({
                'pretrained_file': str(self.FIXTURES_ROOT / 'embeddings/multi-file-archive.zip'),
                'embedding_dim': 5
                })
        with pytest.raises(ValueError, message="No ValueError when pretrained_file is a multi-file archive"):
            Embedding.from_params(vocab, params)

        for ext in ['.zip', '.tar.gz']:
            archive_path = str(self.FIXTURES_ROOT / 'embeddings/multi-file-archive') + ext
            file_uri = format_embeddings_file_uri(archive_path, 'folder/fake_embeddings.5d.txt')
            params = Params({
                    'pretrained_file': file_uri,
                    'embedding_dim': 5
                    })
            embeddings = Embedding.from_params(vocab, params).weight.data
            for tok, vec in token2vec.items():
                i = vocab.get_token_index(tok)
                assert torch.equal(embeddings[i], vec), 'Problem with format ' + archive_path

    def test_embeddings_text_file(self):
        txt_path = str(self.FIXTURES_ROOT / 'utf-8_sample/utf-8_sample.txt')

        # This is for sure a correct way to read an utf-8 encoded text file
        with open(txt_path, 'rt', encoding='utf-8') as f:
            correct_text = f.read()

        # Check if we get the correct text on plain and compressed versions of the file
        paths = [txt_path] + [txt_path + ext for ext in ['.gz', '.zip']]
        for path in paths:
            with EmbeddingsTextFile(path) as f:
                text = f.read()
            assert text == correct_text, "Test failed for file: " + path

        # Check for a file contained inside an archive with multiple files
        for ext in ['.zip', '.tar.gz', '.tar.bz2', '.tar.lzma']:
            archive_path = str(self.FIXTURES_ROOT / 'utf-8_sample/archives/utf-8') + ext
            file_uri = format_embeddings_file_uri(archive_path, 'folder/utf-8_sample.txt')
            with EmbeddingsTextFile(file_uri) as f:
                text = f.read()
            assert text == correct_text, "Test failed for file: " + archive_path

        # Passing a second level path when not reading an archive
        with pytest.raises(ValueError):
            with EmbeddingsTextFile(format_embeddings_file_uri(txt_path, 'a/fake/path')):
                pass

    def test_embeddings_text_file_num_tokens(self):
        test_filename = str(self.TEST_DIR / 'temp_embeddings.vec')

        def check_num_tokens(first_line, expected_num_tokens):
            with open(test_filename, 'w') as f:
                f.write(first_line)
            with EmbeddingsTextFile(test_filename) as f:
                assert f.num_tokens == expected_num_tokens, f"Wrong num tokens for line: {first_line}"

        valid_header_lines = ['1000000 300', '300 1000000', '1000000']
        for line in valid_header_lines:
            check_num_tokens(line, expected_num_tokens=1_000_000)

        not_header_lines = ['hello 1', 'hello 1 2', '111 222 333', '111 222 hello']
        for line in not_header_lines:
            check_num_tokens(line, expected_num_tokens=None)

    def test_decode_embeddings_file_uri(self):
        first_level_paths = [
                'path/to/embeddings.gz',
                'unicode/path/òàè+ù.vec',
                'http://www.embeddings.com/path/to/embeddings.gz',
                'http://www.embeddings.com/àèìòù?query=blabla.zip',
                ]
        second_level_paths = [
                'path/to/glove.27B.300d.vec',
                'òàè+ù.vec',
                'crawl-300d-2M.vec'
                ]

        for simple_path in first_level_paths:
            assert parse_embeddings_file_uri(simple_path) == (simple_path, None)

        for path1, path2 in zip(first_level_paths, second_level_paths):
            uri = format_embeddings_file_uri(path1, path2)
            decoded = parse_embeddings_file_uri(uri)
            assert decoded == (path1, path2)
