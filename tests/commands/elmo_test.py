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
    def test_embedding_works(self):
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
                    "--options-file",
                    self.options_file,
                    "--weight-file",
                    self.weight_file]

        main()

        assert os.path.exists(output_path)

        with h5py.File(output_path, 'r') as h5py_file:
            assert list(h5py_file.keys()) == ["0"]
            # The vectors in the test configuration are smaller (32 length)
            assert h5py_file.get("0").shape == (3, len(sentence.split()), 32)

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
                    "--options-file",
                    self.options_file,
                    "--weight-file",
                    self.weight_file]

        main()

        assert os.path.exists(output_path)

        with h5py.File(output_path, 'r') as h5py_file:
            assert list(h5py_file.keys()) == ["0", "1"]
            # The vectors in the test configuration are smaller (32 length)
            for i, sentence in enumerate(sentences):
                assert h5py_file.get(str(i)).shape == (3, len(sentence.split()), 32)

    def test_batch_embedding_works_with_sentence_key(self):
        tempdir = tempfile.mkdtemp()
        sentences_path = os.path.join(tempdir, "sentences.txt")
        output_path = os.path.join(tempdir, "output.txt")

        sentences = [
                "A Michael went to the store to buy some eggs .",
                "B Joel rolled down the street on his skateboard ."
        ]

        with open(sentences_path, 'w') as f:
            for line in sentences:
                f.write(line + '\n')

        sys.argv = ["run.py",  # executable
                    "elmo",  # command
                    sentences_path,
                    output_path,
                    "--options-file",
                    self.options_file,
                    "--weight-file",
                    self.weight_file,
                    "--use-sentence-key"]

        main()

        assert os.path.exists(output_path)

        with h5py.File(output_path, 'r') as h5py_file:
            assert list(h5py_file.keys()) == ["A", "B"]
            # The vectors in the test configuration are smaller (32 length)
            assert h5py_file.get("A").shape == (3, len(sentences[0].split()) - 1, 32)
            assert h5py_file.get("B").shape == (3, len(sentences[1].split()) - 1, 32)


class TestElmoEmbedder(ElmoTestCase):
    def test_embeddings_are_as_expected(self):
        """
        You can recreate the expected embeddings with the following command:

        python -m allennlp.run elmo \
            tests/fixtures/elmo/sentences.txt \
            tests/fixtures/expected_embeddings.hdf5 \
            --options-file tests/fixtures/elmo/options.json \
            --weight-file tests/fixtures/elmo/lm_weights.hdf5 \
            --batch-size 10
        """
        sentences = []
        with open(self.sentences_txt_file) as fin:
            for line in fin:
                sentences.append(line.split())

        embedder = ElmoEmbedder(options_file=self.options_file, weight_file=self.weight_file)
        embeddings = embedder.embed_sentences(sentences, batch_size=10)

        with h5py.File(self.expected_embeddings_file, "r") as expected:
            for i, tensor in enumerate(embeddings):
                for layer in range(3):
                    expected_tensor = expected.get(str(i))
                    numpy.testing.assert_array_almost_equal(tensor[layer], expected_tensor[layer])

            main()
