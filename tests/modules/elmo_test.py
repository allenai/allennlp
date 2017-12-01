# pylint: disable=no-self-use,invalid-name,protected-access
import os
import json

import h5py
import numpy
import torch
from torch.autograd import Variable

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data import Token, Vocabulary, Dataset, Instance
from allennlp.data.iterators import BasicIterator
from allennlp.modules.elmo import _ElmoBiLm, Elmo, _ElmoCharacterEncoder
from allennlp.data.fields import TextField
from allennlp.nn.util import remove_sentence_boundaries

FIXTURES = os.path.join('tests', 'fixtures', 'elmo')


class TestElmoBiLm(AllenNlpTestCase):
    def test_elmo_bilm(self):
        # get the raw data
        sentences, expected_lm_embeddings = self._load_sentences_embeddings()

        # load the test model
        options_file = os.path.join(FIXTURES, 'options.json')
        weight_file = os.path.join(FIXTURES, 'lm_weights.hdf5')
        elmo_bilm = _ElmoBiLm(options_file, weight_file)

        # Deal with the data.
        indexer = ELMoTokenCharactersIndexer()

        # For each sentence, first create a TextField, then create an instance
        instances = []
        for batch in zip(*sentences):
            for sentence in batch:
                tokens = [Token(token) for token in sentence.split()]
                field = TextField(tokens, {'character_ids': indexer})
                instance = Instance({"elmo": field})
                instances.append(instance)

        dataset = Dataset(instances)
        vocab = Vocabulary()
        dataset.index_instances(vocab)

        # Now finally we can iterate through batches.
        iterator = BasicIterator(3)
        for i, batch in enumerate(iterator(dataset, num_epochs=1, shuffle=False)):
            batch_tensor = Variable(torch.from_numpy(batch['elmo']['character_ids']))
            lm_embeddings = elmo_bilm(batch_tensor)
            top_layer_embeddings, mask = remove_sentence_boundaries(
                    lm_embeddings['activations'][2],
                    lm_embeddings['mask']
            )

            # check the mask lengths
            lengths = mask.data.numpy().sum(axis=1)
            batch_sentences = [sentences[k][i] for k in range(3)]
            expected_lengths = [
                    len(sentence.split()) for sentence in batch_sentences
            ]
            self.assertEqual(lengths.tolist(), expected_lengths)

            # get the expected embeddings and compare!
            expected_top_layer = [expected_lm_embeddings[k][i] for k in range(3)]
            for k in range(3):
                self.assertTrue(
                        numpy.allclose(
                                top_layer_embeddings[k, :lengths[k], :].data.numpy(),
                                expected_top_layer[k],
                                atol=1.0e-6
                        )
                )

    def _load_sentences_embeddings(self):
        # load the test sentences and the expected LM embeddings
        with open(os.path.join(FIXTURES, 'sentences.json')) as fin:
            sentences = json.load(fin)

        # the expected embeddings
        expected_lm_embeddings = []
        for k in range(len(sentences)):
            embed_fname = os.path.join(
                    FIXTURES, 'lm_embeddings_{}.hdf5'.format(k)
            )
            expected_lm_embeddings.append([])
            with h5py.File(embed_fname, 'r') as fin:
                for i in range(10):
                    sent_embeds = fin['%s' % i][...]
                    sent_embeds_concat = numpy.concatenate(
                            (sent_embeds[0, :, :], sent_embeds[1, :, :]),
                            axis=-1
                    )
                    expected_lm_embeddings[-1].append(sent_embeds_concat)

        return sentences, expected_lm_embeddings


class TestElmo(AllenNlpTestCase):
    def test_elmo(self):
        # load the test model
        options_file = os.path.join(FIXTURES, 'options.json')
        weight_file = os.path.join(FIXTURES, 'lm_weights.hdf5')
        elmo = Elmo(options_file, weight_file, 2)

        # Correctness checks are in ElmoBiLm and ScalarMix, here we just add a shallow test
        # to ensure things execute.
        indexer = ELMoTokenCharactersIndexer()
        sentences = [['The', 'sentence', '.'],
                     ['ELMo', 'helps', 'disambiguate', 'ELMo', 'from', 'Elmo', '.']]

        # For each sentence, first create a TextField, then create an instance
        instances = []
        for sentence in sentences:
            tokens = [Token(token) for token in sentence]
            field = TextField(tokens, {'character_ids': indexer})
            instance = Instance({'elmo': field})
            instances.append(instance)

        dataset = Dataset(instances)
        vocab = Vocabulary()
        dataset.index_instances(vocab)
        character_ids = dataset.as_array_dict()['elmo']['character_ids']

        output = elmo(Variable(torch.from_numpy(character_ids)))
        elmo_representations = output['elmo_representations']
        mask = output['mask']

        assert len(elmo_representations) == 2
        assert list(elmo_representations[0].size()) == [2, 7, 32]
        assert list(elmo_representations[1].size()) == [2, 7, 32]
        assert list(mask.size()) == [2, 7]


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

        elmo_token_embedder = _ElmoCharacterEncoder(options_file, weight_file)
        elmo_token_embedder_output = elmo_token_embedder(batch)

        # Reshape back to a list of words and compare with ground truth.  Need to also
        # remove <S>, </S>
        actual_embeddings = remove_sentence_boundaries(
                elmo_token_embedder_output['token_embedding'],
                elmo_token_embedder_output['mask']
        )[0].data.numpy()
        actual_embeddings = actual_embeddings.reshape(-1, actual_embeddings.shape[-1])

        embedding_file = os.path.join(FIXTURES, 'elmo_token_embeddings.hdf5')
        with h5py.File(embedding_file, 'r') as fin:
            expected_embeddings = fin['embedding'][...]

        assert numpy.allclose(actual_embeddings[:len(tokens)], expected_embeddings, atol=1e-6)

    def test_elmo_token_representation_bos_eos(self):
        # The additional <S> and </S> embeddings added by the embedder should be as expected.
        indexer = ELMoTokenCharactersIndexer()

        options_file = os.path.join(FIXTURES, 'options.json')
        weight_file = os.path.join(FIXTURES, 'lm_weights.hdf5')

        elmo_token_embedder = _ElmoCharacterEncoder(options_file, weight_file)

        for correct_index, token in [[0, '<S>'], [2, '</S>']]:
            indices = indexer.token_to_indices(Token(token), Vocabulary())
            indices = Variable(torch.from_numpy(numpy.array(indices))).view(1, 1, -1)
            embeddings = elmo_token_embedder(indices)['token_embedding']
            assert numpy.allclose(embeddings[0, correct_index, :].data.numpy(), embeddings[0, 1, :].data.numpy())
