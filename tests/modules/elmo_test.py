# pylint: disable=no-self-use,invalid-name,protected-access
import os
import h5py
import json

import numpy
import torch
from torch.autograd import Variable

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.modules.elmo import _ElmoBiLm, Elmo
from allennlp.data.fields import TextField
from allennlp.data import Token, Vocabulary, Dataset, Instance
from allennlp.data.iterators import BasicIterator

FIXTURES = os.path.join('tests', 'fixtures', 'elmo')


class TestElmoBiLm(AllenNlpTestCase):
    def test_elmo_bilm(self):
        def _load_sentences_embeddings():
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

        # get the raw data
        sentences, expected_lm_embeddings = _load_sentences_embeddings()

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
            top_layer_embeddings = lm_embeddings['activations'][2].data.numpy()

            # check the mask lengths
            lengths = lm_embeddings['mask'].data.numpy().sum(axis=1)
            batch_sentences = [sentences[k][i] for k in range(3)]
            expected_lengths = [
                    # + 2 for <S> and </S>
                    len(sentence.split()) + 2 for sentence in batch_sentences
            ]
            self.assertEqual(lengths.tolist(), expected_lengths)

            # get the expected embeddings and compare!
            expected_top_layer = [expected_lm_embeddings[k][i] for k in range(3)]
            for k in range(3):
                self.assertTrue(
                    numpy.allclose(
                        top_layer_embeddings[k, 1:(lengths[k] - 1), :],
                        expected_top_layer[k],
                        atol=1.0e-6
                    )
                )

class TestElmo(AllenNlpTestCase):
    def test_elmo(self):
        # load the test model
        options_file = os.path.join(FIXTURES, 'options.json')
        weight_file = os.path.join(FIXTURES, 'lm_weights.hdf5')
        elmo = Elmo(options_file, weight_file, 2)

        # Correctness checks are elsewhere, here we just add a shallow test
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
        elmo_representations = output['elmo']
        mask = output['mask']

        assert len(elmo_representations) == 2
        assert list(elmo_representations[0].size()) == [2, 7, 32]
        assert list(elmo_representations[1].size()) == [2, 7, 32]
        assert list(mask.size()) == [2, 7]
