# pylint: disable=no-self-use,invalid-name
import os
import pathlib
import sys
import tempfile

import h5py
import numpy

from allennlp.commands import main
from allennlp.commands.elmo import ElmoEmbedder
from allennlp.tests.modules.elmo_test import ElmoTestCase


class TestElmoCommand(ElmoTestCase):
    def setUp(self):
        super(TestElmoCommand, self).setUp()
        self.tempdir = pathlib.Path(tempfile.mkdtemp())
        self.sentences_path = str(self.tempdir / "sentences.txt")
        self.output_path = str(self.tempdir / "output.txt")

    def test_all_embedding_works(self):
        sentence = "Michael went to the store to buy some eggs ."
        with open(self.sentences_path, 'w') as f:
            f.write(sentence)

        sys.argv = ["run.py",  # executable
                    "elmo",  # command
                    self.sentences_path,
                    self.output_path,
                    "--all",
                    "--options-file",
                    self.options_file,
                    "--weight-file",
                    self.weight_file]

        main()

        assert os.path.exists(self.output_path)

        embedder = ElmoEmbedder(options_file=self.options_file, weight_file=self.weight_file)
        expected_embedding = embedder.embed_sentence(sentence.split())

        with h5py.File(self.output_path, 'r') as h5py_file:
            assert list(h5py_file.keys()) == [sentence]
            # The vectors in the test configuration are smaller (32 length)
            embedding = h5py_file.get(sentence)
            assert embedding.shape == (3, len(sentence.split()), 32)
            numpy.testing.assert_allclose(embedding, expected_embedding, rtol=1e-4)

    def test_top_embedding_works(self):
        sentence = "Michael went to the store to buy some eggs ."
        with open(self.sentences_path, 'w') as f:
            f.write(sentence)

        sys.argv = ["run.py",  # executable
                    "elmo",  # command
                    self.sentences_path,
                    self.output_path,
                    "--top",
                    "--options-file",
                    self.options_file,
                    "--weight-file",
                    self.weight_file]

        main()

        assert os.path.exists(self.output_path)

        embedder = ElmoEmbedder(options_file=self.options_file, weight_file=self.weight_file)
        expected_embedding = embedder.embed_sentence(sentence.split())[2]

        with h5py.File(self.output_path, 'r') as h5py_file:
            assert list(h5py_file.keys()) == [sentence]
            # The vectors in the test configuration are smaller (32 length)
            embedding = h5py_file.get(sentence)
            assert embedding.shape == (len(sentence.split()), 32)
            numpy.testing.assert_allclose(embedding, expected_embedding, rtol=1e-4)

    def test_average_embedding_works(self):
        sentence = "Michael went to the store to buy some eggs ."
        with open(self.sentences_path, 'w') as f:
            f.write(sentence)

        sys.argv = ["run.py",  # executable
                    "elmo",  # command
                    self.sentences_path,
                    self.output_path,
                    "--average",
                    "--options-file",
                    self.options_file,
                    "--weight-file",
                    self.weight_file]

        main()

        assert os.path.exists(self.output_path)

        embedder = ElmoEmbedder(options_file=self.options_file, weight_file=self.weight_file)
        expected_embedding = embedder.embed_sentence(sentence.split())
        expected_embedding = (expected_embedding[0] + expected_embedding[1] + expected_embedding[2]) / 3

        with h5py.File(self.output_path, 'r') as h5py_file:
            assert list(h5py_file.keys()) == [sentence]
            # The vectors in the test configuration are smaller (32 length)
            embedding = h5py_file.get(sentence)
            assert embedding.shape == (len(sentence.split()), 32)
            numpy.testing.assert_allclose(embedding, expected_embedding, rtol=1e-4)

    def test_batch_embedding_works(self):
        sentences = [
                "Michael went to the store to buy some eggs .",
                "Joel rolled down the street on his skateboard ."
        ]

        with open(self.sentences_path, 'w') as f:
            for line in sentences:
                f.write(line + '\n')

        sys.argv = ["run.py",  # executable
                    "elmo",  # command
                    self.sentences_path,
                    self.output_path,
                    "--all",
                    "--options-file",
                    self.options_file,
                    "--weight-file",
                    self.weight_file]

        main()

        assert os.path.exists(self.output_path)

        with h5py.File(self.output_path, 'r') as h5py_file:
            assert set(h5py_file.keys()) == set(sentences)
            # The vectors in the test configuration are smaller (32 length)
            for sentence in sentences:
                assert h5py_file.get(sentence).shape == (3, len(sentence.split()), 32)

    def test_duplicate_sentences(self):
        sentences = [
                "Michael went to the store to buy some eggs .",
                "Michael went to the store to buy some eggs .",
        ]

        with open(self.sentences_path, 'w') as f:
            for line in sentences:
                f.write(line + '\n')

        sys.argv = ["run.py",  # executable
                    "elmo",  # command
                    self.sentences_path,
                    self.output_path,
                    "--all",
                    "--options-file",
                    self.options_file,
                    "--weight-file",
                    self.weight_file]

        main()

        assert os.path.exists(self.output_path)

        with h5py.File(self.output_path, 'r') as h5py_file:
            assert len(h5py_file.keys()) == 1
            assert set(h5py_file.keys()) == set(sentences)
            # The vectors in the test configuration are smaller (32 length)
            for sentence in set(sentences):
                assert h5py_file.get(sentence).shape == (3, len(sentence.split()), 32)

    def test_empty_sentences_are_filtered(self):
        sentences = [
                "A",
                "",
                "",
                "B"
        ]

        with open(self.sentences_path, 'w') as f:
            for line in sentences:
                f.write(line + '\n')

        sys.argv = ["run.py",  # executable
                    "elmo",  # command
                    self.sentences_path,
                    self.output_path,
                    "--all",
                    "--options-file",
                    self.options_file,
                    "--weight-file",
                    self.weight_file]

        main()

        assert os.path.exists(self.output_path)

        with h5py.File(self.output_path, 'r') as h5py_file:
            assert len(h5py_file.keys()) == 2
            assert set(h5py_file.keys()) == set(["A", "B"])


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

    def test_embed_batch_is_empty_sentence(self):
        embedder = ElmoEmbedder(options_file=self.options_file, weight_file=self.weight_file)
        embeddings = embedder.embed_sentence([])

        assert embeddings.shape == (3, 0, 1024)

    def test_embed_batch_contains_empty_sentence(self):
        embedder = ElmoEmbedder(options_file=self.options_file, weight_file=self.weight_file)
        embeddings = list(embedder.embed_sentences(["This is a test".split(), []]))

        assert len(embeddings) == 2
