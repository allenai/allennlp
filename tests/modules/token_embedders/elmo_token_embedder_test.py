# pylint: disable=no-self-use

import os
import numpy
import h5py

import torch
from torch.autograd import Variable


from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.token_embedders.elmo_token_embedder import _ElmoTokenRepresentation
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data import Token, Vocabulary

FIXTURES = os.path.join('tests', 'fixtures', 'elmo')


class TestElmoTokenRepresentation(AllenNlpTestCase):
    def test_elmo_token_representation(self):
        # Load the test words and convert to char ids
        with open(os.path.join(FIXTURES, 'vocab_test.txt'), 'r') as fin:
            tokens = fin.read().strip().split('\n')

        indexer = ELMoTokenCharactersIndexer()
        indices = [indexer.token_to_indices(Token(token), Vocabulary()) for token in tokens]
        # There are 457 tokens. Reshape into 10 batches of 50 tokens.
        sentences = []
        for k in range(10):
            sentences.append(
                    indexer.pad_token_sequence(
                            indices[(k * 50):((k + 1) * 50)], desired_num_tokens=50, padding_lengths={}
                    )
            )
        batch = Variable(torch.from_numpy(numpy.array(sentences)))

        options_file = os.path.join(FIXTURES, 'options.json')
        weight_file = os.path.join(FIXTURES, 'lm_weights.hdf5')

        elmo_token_embedder = _ElmoTokenRepresentation(options_file, weight_file)
        token_embedding = elmo_token_embedder(batch)['token_embedding'].data.numpy()

        # Reshape back to a list of words and compare with ground truth.  Need to also
        # remove <S>, </S>
        actual_embeddings = token_embedding[:, 1:-1, :].reshape(-1, token_embedding.shape[-1])

        embedding_file = os.path.join(FIXTURES, 'elmo_token_embeddings.hdf5')
        with h5py.File(embedding_file, 'r') as fin:
            expected_embeddings = fin['embedding'][...]

        assert numpy.allclose(actual_embeddings[:len(tokens)], expected_embeddings, atol=1e-6)

    def test_elmo_token_representation_bos_eos(self):
        # The additional <S> and </S> embeddings added by the embedder should be as expected.
        indexer = ELMoTokenCharactersIndexer()

        options_file = os.path.join(FIXTURES, 'options.json')
        weight_file = os.path.join(FIXTURES, 'lm_weights.hdf5')

        elmo_token_embedder = _ElmoTokenRepresentation(options_file, weight_file)

        # First <S>
        for correct_index, token in [[0, '<S>'], [2, '</S>']]:
            indices = indexer.token_to_indices(Token(token), Vocabulary())
            indices = Variable(torch.from_numpy(numpy.array(indices))).view(1, 1, -1)
            embeddings = elmo_token_embedder(indices)['token_embedding']
            assert numpy.allclose(embeddings[0, correct_index, :].data.numpy(), embeddings[0, 1, :].data.numpy())
