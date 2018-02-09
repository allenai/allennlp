# pylint: disable=no-self-use,invalid-name
import os
import sys
import tempfile

import h5py
import numpy

from allennlp.commands import main
from allennlp.commands.elmo import ElmoEmbedder
from tests.modules.elmo_test import ElmoTestCase


class TestElmoCommand(ElmoTestCase):
    def test_all_embedding_works(self):
        tempdir = tempfile.mkdtemp()
        sentences_path = os.path.join(tempdir, "sentences.txt")
        output_path = os.path.join(tempdir, "output.txt")

        sentence = "Michael went to the store to buy some eggs ."
        with open(sentences_path, 'w') as f:
            f.write(sentence)

        sys.argv = ["run.py",  # executable
                    "elmo",  # command
                    sentences_path,
                    output_path,
                    "--all",
                    "--options-file",
                    self.options_file,
                    "--weight-file",
                    self.weight_file]

        main()

        assert os.path.exists(output_path)

        embedder = ElmoEmbedder(options_file=self.options_file, weight_file=self.weight_file)
        expected_embedding = embedder.embed_sentence(sentence.split())

        with h5py.File(output_path, 'r') as h5py_file:
            assert list(h5py_file.keys()) == [sentence]
            # The vectors in the test configuration are smaller (32 length)
            embedding = h5py_file.get(sentence)
            assert embedding.shape == (3, len(sentence.split()), 32)
            numpy.testing.assert_allclose(embedding, expected_embedding, rtol=1e-4)

    def test_top_embedding_works(self):
        tempdir = tempfile.mkdtemp()
        sentences_path = os.path.join(tempdir, "sentences.txt")
        output_path = os.path.join(tempdir, "output.txt")

        sentence = "Michael went to the store to buy some eggs ."
        with open(sentences_path, 'w') as f:
            f.write(sentence)

        sys.argv = ["run.py",  # executable
                    "elmo",  # command
                    sentences_path,
                    output_path,
                    "--top",
                    "--options-file",
                    self.options_file,
                    "--weight-file",
                    self.weight_file]

        main()

        assert os.path.exists(output_path)

        embedder = ElmoEmbedder(options_file=self.options_file, weight_file=self.weight_file)
        expected_embedding = embedder.embed_sentence(sentence.split())[2]

        with h5py.File(output_path, 'r') as h5py_file:
            assert list(h5py_file.keys()) == [sentence]
            # The vectors in the test configuration are smaller (32 length)
            embedding = h5py_file.get(sentence)
            assert embedding.shape == (len(sentence.split()), 32)
            numpy.testing.assert_allclose(embedding, expected_embedding, rtol=1e-4)

    def test_average_embedding_works(self):
        tempdir = tempfile.mkdtemp()
        sentences_path = os.path.join(tempdir, "sentences.txt")
        output_path = os.path.join(tempdir, "output.txt")

        sentence = "Michael went to the store to buy some eggs ."
        with open(sentences_path, 'w') as f:
            f.write(sentence)

        sys.argv = ["run.py",  # executable
                    "elmo",  # command
                    sentences_path,
                    output_path,
                    "--average",
                    "--options-file",
                    self.options_file,
                    "--weight-file",
                    self.weight_file]

        main()

        assert os.path.exists(output_path)

        embedder = ElmoEmbedder(options_file=self.options_file, weight_file=self.weight_file)
        expected_embedding = embedder.embed_sentence(sentence.split())
        expected_embedding = (expected_embedding[0] + expected_embedding[1] + expected_embedding[2]) / 3

        with h5py.File(output_path, 'r') as h5py_file:
            assert list(h5py_file.keys()) == [sentence]
            # The vectors in the test configuration are smaller (32 length)
            embedding = h5py_file.get(sentence)
            assert embedding.shape == (len(sentence.split()), 32)
            numpy.testing.assert_allclose(embedding, expected_embedding, rtol=1e-4)

    def test_batch_embedding_works(self):
        tempdir = tempfile.mkdtemp()
        sentences_path = os.path.join(tempdir, "sentences.txt")
        output_path = os.path.join(tempdir, "output.txt")

        sentences = [
                "Michael went to the store to buy some eggs .",
                "Joel rolled down the street on his skateboard ."
        ]

        with open(sentences_path, 'w') as f:
            for line in sentences:
                f.write(line + '\n')

        sys.argv = ["run.py",  # executable
                    "elmo",  # command
                    sentences_path,
                    output_path,
                    "--all",
                    "--options-file",
                    self.options_file,
                    "--weight-file",
                    self.weight_file]

        main()

        assert os.path.exists(output_path)

        with h5py.File(output_path, 'r') as h5py_file:
            assert set(h5py_file.keys()) == set(sentences)
            # The vectors in the test configuration are smaller (32 length)
            for sentence in sentences:
                assert h5py_file.get(sentence).shape == (3, len(sentence.split()), 32)

    def test_duplicate_sentences(self):
        tempdir = tempfile.mkdtemp()
        sentences_path = os.path.join(tempdir, "sentences.txt")
        output_path = os.path.join(tempdir, "output.txt")

        sentences = [
                "Michael went to the store to buy some eggs .",
                "Michael went to the store to buy some eggs .",
        ]

        with open(sentences_path, 'w') as f:
            for line in sentences:
                f.write(line + '\n')

        sys.argv = ["run.py",  # executable
                    "elmo",  # command
                    sentences_path,
                    output_path,
                    "--all",
                    "--options-file",
                    self.options_file,
                    "--weight-file",
                    self.weight_file]

        main()

        assert os.path.exists(output_path)

        with h5py.File(output_path, 'r') as h5py_file:
            assert len(h5py_file.keys()) == 1
            assert set(h5py_file.keys()) == set(sentences)
            # The vectors in the test configuration are smaller (32 length)
            for sentence in set(sentences):
                assert h5py_file.get(sentence).shape == (3, len(sentence.split()), 32)


class TestElmoEmbedder(ElmoTestCase):
    def test_embeddings_are_as_expected(self):
        loaded_sentences, loaded_embeddings = self._load_sentences_embeddings()

        assert len(loaded_sentences) == len(loaded_embeddings)
        batch_size = len(loaded_sentences)

        # The sentences and embeddings are organized in an idiosyncratic way TensorFlow handles batching.
        # We are going to reorganize them linearly so they can be grouped into batches by AllenNLP.
        sentences = []
        expected_embeddings = []
        for batch_number in range(len(loaded_sentences[0])):
            for index in range(batch_size):
                sentences.append(loaded_sentences[index][batch_number].split())
                expected_embeddings.append(loaded_embeddings[index][batch_number])

        assert len(expected_embeddings) == len(sentences)

        embedder = ElmoEmbedder(options_file=self.options_file, weight_file=self.weight_file)
        embeddings = list(embedder.embed_sentences(sentences, batch_size))

        assert len(embeddings) == len(sentences)

        for tensor, expected in zip(embeddings, expected_embeddings):
            numpy.testing.assert_array_almost_equal(tensor[2], expected)
