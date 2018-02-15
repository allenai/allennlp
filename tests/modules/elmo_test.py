# pylint: disable=no-self-use,invalid-name,protected-access
import os
import json

import h5py
import numpy
import torch
from torch.autograd import Variable

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data import Token, Vocabulary, Instance
from allennlp.data.dataset import Batch
from allennlp.data.iterators import BasicIterator
from allennlp.modules.elmo import _ElmoBiLm, Elmo, _ElmoCharacterEncoder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.data.fields import TextField
from allennlp.nn.util import remove_sentence_boundaries


class ElmoTestCase(AllenNlpTestCase):
    def setUp(self):
        super(ElmoTestCase, self).setUp()
        self.fixtures_path = os.path.join('tests', 'fixtures', 'elmo')
        self.options_file = os.path.join(self.fixtures_path, 'options.json')
        self.weight_file = os.path.join(self.fixtures_path, 'lm_weights.hdf5')
        self.sentences_json_file = os.path.join(self.fixtures_path, 'sentences.json')
        self.sentences_txt_file = os.path.join(self.fixtures_path, 'sentences.txt')

    def _load_sentences_embeddings(self):
        """
        Load the test sentences and the expected LM embeddings.

        These files loaded in this method were created with a batch-size of 3.
        Due to idiosyncrasies with TensorFlow, the 30 sentences in sentences.json are split into 3 files in which
        the k-th sentence in each is from batch k.

        This method returns a (sentences, embeddings) pair where each is a list of length batch_size.
        Each list contains a sublist with total_sentence_count / batch_size elements.  As with the original files,
        the k-th element in the sublist is in batch k.
        """
        with open(self.sentences_json_file) as fin:
            sentences = json.load(fin)

        # the expected embeddings
        expected_lm_embeddings = []
        for k in range(len(sentences)):
            embed_fname = os.path.join(
                    self.fixtures_path, 'lm_embeddings_{}.hdf5'.format(k)
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


class TestElmoBiLm(ElmoTestCase):
    def test_elmo_bilm(self):
        # get the raw data
        sentences, expected_lm_embeddings = self._load_sentences_embeddings()

        # load the test model
        elmo_bilm = _ElmoBiLm(self.options_file, self.weight_file)

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

        vocab = Vocabulary()

        # Now finally we can iterate through batches.
        iterator = BasicIterator(3)
        iterator.index_with(vocab)
        for i, batch in enumerate(iterator(instances, num_epochs=1, shuffle=False)):
            lm_embeddings = elmo_bilm(batch['elmo']['character_ids'])
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


class TestElmo(ElmoTestCase):
    def setUp(self):
        super(TestElmo, self).setUp()

        self.elmo = Elmo(self.options_file, self.weight_file, 2, dropout=0.0)

    def _sentences_to_ids(self, sentences):
        indexer = ELMoTokenCharactersIndexer()

        # For each sentence, first create a TextField, then create an instance
        instances = []
        for sentence in sentences:
            tokens = [Token(token) for token in sentence]
            field = TextField(tokens, {'character_ids': indexer})
            instance = Instance({'elmo': field})
            instances.append(instance)

        dataset = Batch(instances)
        vocab = Vocabulary()
        dataset.index_instances(vocab)
        return dataset.as_tensor_dict()['elmo']['character_ids']

    def test_elmo(self):
        # Correctness checks are in ElmoBiLm and ScalarMix, here we just add a shallow test
        # to ensure things execute.
        sentences = [['The', 'sentence', '.'],
                     ['ELMo', 'helps', 'disambiguate', 'ELMo', 'from', 'Elmo', '.']]

        character_ids = self._sentences_to_ids(sentences)
        output = self.elmo(character_ids)
        elmo_representations = output['elmo_representations']
        mask = output['mask']

        assert len(elmo_representations) == 2
        assert list(elmo_representations[0].size()) == [2, 7, 32]
        assert list(elmo_representations[1].size()) == [2, 7, 32]
        assert list(mask.size()) == [2, 7]

    def test_elmo_4D_input(self):
        sentences = [[['The', 'sentence', '.'],
                      ['ELMo', 'helps', 'disambiguate', 'ELMo', 'from', 'Elmo', '.']],
                     [['1', '2'], ['1', '2', '3', '4', '5', '6', '7']],
                     [['1', '2', '3', '4', '50', '60', '70'], ['The']]]

        all_character_ids = []
        for batch_sentences in sentences:
            all_character_ids.append(self._sentences_to_ids(batch_sentences))

        # (2, 3, 7, 50)
        character_ids = torch.cat([ids.unsqueeze(1) for ids in all_character_ids], dim=1)
        embeddings_4d = self.elmo(character_ids)

        # Run the individual batches.
        embeddings_3d = []
        for char_ids in all_character_ids:
            self.elmo._elmo_lstm._elmo_lstm.reset_states()
            embeddings_3d.append(self.elmo(char_ids))

        for k in range(3):
            numpy.testing.assert_array_almost_equal(
                    embeddings_4d['elmo_representations'][0][:, k, :, :].data.numpy(),
                    embeddings_3d[k]['elmo_representations'][0].data.numpy()
            )


class TestElmoRequiresGrad(ElmoTestCase):
    def _run_test(self, requires_grad):
        embedder = ElmoTokenEmbedder(self.options_file, self.weight_file, requires_grad=requires_grad)
        batch_size = 3
        seq_len = 4
        char_ids = Variable(torch.from_numpy(numpy.random.randint(0, 262, (batch_size, seq_len, 50))))
        embeddings = embedder(char_ids)
        loss = embeddings.sum()
        loss.backward()

        elmo_grads = [param.grad for name, param in embedder.named_parameters() if '_elmo_lstm' in name]
        if requires_grad:
            # None of the elmo grads should be None.
            assert all([grad is not None for grad in elmo_grads])
        else:
            # All of the elmo grads should be None.
            assert all([grad is None for grad in elmo_grads])

    def test_elmo_requires_grad(self):
        self._run_test(True)

    def test_elmo_does_not_require_grad(self):
        self._run_test(False)


class TestElmoTokenRepresentation(ElmoTestCase):
    def test_elmo_token_representation(self):
        # Load the test words and convert to char ids
        with open(os.path.join(self.fixtures_path, 'vocab_test.txt'), 'r') as fin:
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

        elmo_token_embedder = _ElmoCharacterEncoder(self.options_file, self.weight_file)
        elmo_token_embedder_output = elmo_token_embedder(batch)

        # Reshape back to a list of words and compare with ground truth.  Need to also
        # remove <S>, </S>
        actual_embeddings = remove_sentence_boundaries(
                elmo_token_embedder_output['token_embedding'],
                elmo_token_embedder_output['mask']
        )[0].data.numpy()
        actual_embeddings = actual_embeddings.reshape(-1, actual_embeddings.shape[-1])

        embedding_file = os.path.join(self.fixtures_path, 'elmo_token_embeddings.hdf5')
        with h5py.File(embedding_file, 'r') as fin:
            expected_embeddings = fin['embedding'][...]

        assert numpy.allclose(actual_embeddings[:len(tokens)], expected_embeddings, atol=1e-6)

    def test_elmo_token_representation_bos_eos(self):
        # The additional <S> and </S> embeddings added by the embedder should be as expected.
        indexer = ELMoTokenCharactersIndexer()

        elmo_token_embedder = _ElmoCharacterEncoder(self.options_file, self.weight_file)

        for correct_index, token in [[0, '<S>'], [2, '</S>']]:
            indices = indexer.token_to_indices(Token(token), Vocabulary())
            indices = Variable(torch.from_numpy(numpy.array(indices))).view(1, 1, -1)
            embeddings = elmo_token_embedder(indices)['token_embedding']
            assert numpy.allclose(embeddings[0, correct_index, :].data.numpy(), embeddings[0, 1, :].data.numpy())
