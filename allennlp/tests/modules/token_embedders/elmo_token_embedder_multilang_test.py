# pylint: disable=no-self-use,invalid-name
import filecmp
import json
import os
import pathlib
import tarfile
import numpy as np

import torch

from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.common.testing import ModelTestCase
from allennlp.modules.token_embedders import ElmoTokenEmbedderMultiLang


class TestElmoTokenEmbedderMultilang(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'elmo_multilingual' / 'config' / 'multilang_token_embedder.json',
                          self.FIXTURES_ROOT / 'data' / 'dependencies_multilang/*')

    def test_tagger_with_elmo_token_embedder_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_file_archiving(self):
        # This happens to be a good place to test auxiliary file archiving.
        # Train the model
        params = Params.from_file(self.FIXTURES_ROOT / 'elmo_multilingual' /
                                  'config' / 'multilang_token_embedder.json')
        serialization_dir = os.path.join(self.TEST_DIR, 'serialization')
        train_model(params, serialization_dir)

        # Inspect the archive
        archive_file = os.path.join(serialization_dir, 'model.tar.gz')
        unarchive_dir = os.path.join(self.TEST_DIR, 'unarchive')
        with tarfile.open(archive_file, 'r:gz') as archive:
            archive.extractall(unarchive_dir)

        # It should contain `files_to_archive.json`
        fta_file = os.path.join(unarchive_dir, 'files_to_archive.json')
        assert os.path.exists(fta_file)

        # Which should properly contain { flattened_key -> original_filename }
        with open(fta_file) as fta:
            files_to_archive = json.loads(fta.read())

        assert files_to_archive == {
                'model.text_field_embedder.token_embedders.elmo.options_files.en':
                str(
                        pathlib.Path('allennlp') / 'tests' / 'fixtures' /
                        'elmo' / 'options.json'),
                'model.text_field_embedder.token_embedders.elmo.weight_files.en':
                str(
                        pathlib.Path('allennlp') / 'tests' / 'fixtures' /
                        'elmo' / 'lm_weights.hdf5'),
                'model.text_field_embedder.token_embedders.elmo.options_files.fr':
                str(
                        pathlib.Path('allennlp') / 'tests' / 'fixtures' /
                        'elmo_multilingual' / 'fr' / 'options.json'),
                'model.text_field_embedder.token_embedders.elmo.weight_files.fr':
                str(
                        pathlib.Path('allennlp') / 'tests' / 'fixtures' /
                        'elmo_multilingual' / 'fr' / 'weights.hdf5'),
        }

        # Check that the unarchived contents of those files match the original contents.
        for key, original_filename in files_to_archive.items():
            new_filename = os.path.join(unarchive_dir, "fta", key)
            assert filecmp.cmp(original_filename, new_filename)

    def test_alignment_per_language(self):
        '''
        Tests that the correct alignment is applied for each input language.
        An all-zero matrix is used for English in order to simulate this test.
        '''
        params_dict = {
                'options_files': {
                        'en':
                        self.FIXTURES_ROOT / 'elmo_multilingual' / 'fr' / 'options.json',
                        'fr':
                        self.FIXTURES_ROOT / 'elmo_multilingual' / 'fr' / 'options.json'
                },
                'weight_files': {
                        'en':
                        self.FIXTURES_ROOT / 'elmo_multilingual' / 'fr' / 'weights.hdf5',
                        'fr':
                        self.FIXTURES_ROOT / 'elmo_multilingual' / 'fr' / 'weights.hdf5'
                },
                'aligning_files': {
                        'en':
                        self.FIXTURES_ROOT / 'elmo_multilingual' / 'fr' /  'align_zero.pth',
                        'fr':
                        self.FIXTURES_ROOT / 'elmo_multilingual' / 'fr' / 'align.pth'
                },
                "scalar_mix_parameters": [
                        -9e10,
                        1,
                        -9e10
                ],
        }
        word1 = [0] * 50
        word2 = [0] * 50
        word1[0] = 6
        word1[1] = 5
        word1[2] = 4
        word1[3] = 3
        word2[0] = 3
        word2[1] = 2
        word2[2] = 1
        word2[3] = 0
        embedding_layer = ElmoTokenEmbedderMultiLang.from_params(vocab=None, params=Params(params_dict))

        input_tensor = torch.LongTensor([[word1, word2]])
        embedded_en = embedding_layer(input_tensor, lang='en').data.numpy()
        embedded_fr = embedding_layer(input_tensor, lang='fr').data.numpy()
        assert np.count_nonzero(embedded_en) == 0
        assert np.count_nonzero(embedded_fr) > 0

    def test_elmo_per_language(self):
        '''
        Tests that the correct ELMo weights are applied for each input language.
        The CNN values of the English model are zero in order to simulate the test.
        '''
        params_dict = {
                'options_files': {
                        'en':
                        self.FIXTURES_ROOT / 'elmo_multilingual' / 'fr' / 'options.json',
                        'fr':
                        self.FIXTURES_ROOT / 'elmo_multilingual' / 'fr' / 'options.json'
                },
                'weight_files': {
                        'en':
                        self.FIXTURES_ROOT / 'elmo_multilingual' / 'fr' / 'weights_zero.hdf5',
                        'fr':
                        self.FIXTURES_ROOT / 'elmo_multilingual' / 'fr' / 'weights.hdf5'
                },
                'aligning_files': {
                        'en':
                        self.FIXTURES_ROOT / 'elmo_multilingual' / 'fr' /  'align.pth',
                        'fr':
                        self.FIXTURES_ROOT / 'elmo_multilingual' / 'fr' / 'align.pth'
                },
                "scalar_mix_parameters": [
                        -9e10,
                        1,
                        -9e10
                ],
        }
        word1 = [0] * 50
        word2 = [0] * 50
        word1[0] = 6
        word1[1] = 5
        word1[2] = 4
        word1[3] = 3
        word2[0] = 3
        word2[1] = 2
        word2[2] = 1
        word2[3] = 0
        embedding_layer = ElmoTokenEmbedderMultiLang.from_params(vocab=None, params=Params(params_dict))

        input_tensor = torch.LongTensor([[word1, word2]])
        embedded_en = embedding_layer(input_tensor, lang='en').data.numpy()
        embedded_fr = embedding_layer(input_tensor, lang='fr').data.numpy()
        assert np.count_nonzero(embedded_en) == 0
        assert np.count_nonzero(embedded_fr) > 0
