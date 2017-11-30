# pylint: disable=no-self-use

import os
import numpy
import h5py

import torch
from torch.autograd import Variable
import argparse

from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data import Token, Vocabulary


def main(vocab_path: str, elmo_model_dir: str, output_path: str, device: int):

    # Load the vocabulary words and convert to char ids
    with open(vocab_path, 'r') as vocab_file:
        tokens = vocab_file.read().strip().split('\n')

    indexer = ELMoTokenCharactersIndexer()
    indices = [indexer.token_to_indices(Token(token), Vocabulary()) for token in tokens]
    # There are 457 tokens. Reshape into 10 batches of 50 tokens.
    sentences = []
    for k in range(10):
        sentences.append(indexer.pad_token_sequence(indices[(k * 50):((k + 1) * 50)],
                                                    desired_num_tokens=50,
                                                    padding_lengths={}))

    batch = Variable(torch.from_numpy(numpy.array(sentences)))

    options_file = os.path.join(elmo_model_dir, 'options.json')
    weight_file = os.path.join(elmo_model_dir, 'lm_weights.hdf5')

    elmo_token_embedder = _ElmoTokenRepresentation(options_file, weight_file)
    token_embedding = elmo_token_embedder(batch)['token_embedding'].data.numpy()

    # Reshape back to a list of words and compare with ground truth.  Need to also
    # remove <S>, </S>
    actual_embeddings = token_embedding[:, 1:-1, :].reshape(-1, token_embedding.shape[-1])

    embedding_file = os.path.join(FIXTURES, 'elmo_token_embeddings.hdf5')
    with h5py.File(embedding_file, 'r') as fin:
        expected_embeddings = fin['embedding'][...]

    assert numpy.allclose(actual_embeddings[:len(tokens)], expected_embeddings, atol=1e-6)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate CNN representations for a vocabulary '
                                                 'using ELMo')
    parser.add_argument('--vocab_path', type=str, help='A path to a vocabulary file to generate '
                                                       'representations for.')
    parser.add_argument('--elmo_model_dir', type=str, help='The path to a directory containing an '
                                                           'ELMo config file and weights.')
    parser.add_argument('--output_path', type=str, help='The output path to store the '
                                                        'serialised embeddings.')
    parser.add_argument('--device', type=int, default=-1, help='The device to run on.')

    args = parser.parse_args()
    main(args.vocab_path, args.elmo_model_dir, args.output_path, args.device)