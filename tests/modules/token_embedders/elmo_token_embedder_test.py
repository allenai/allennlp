# pylint: disable=no-self-use,invalid-name
import filecmp
import json
import os
import tarfile

from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch

class TestElmoTokenEmbedder(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model('tests/fixtures/elmo/config/characters_token_embedder.json',
                          'tests/fixtures/data/conll2003.txt')

    def test_tagger_with_elmo_token_embedder_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_tagger_with_elmo_token_embedder_forward_pass_runs_correctly(self):
        dataset = Batch(self.instances)
        dataset.index_instances(self.vocab)
        training_tensors = dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        tags = output_dict['tags']
        assert len(tags) == 2
        assert len(tags[0]) == 7
        assert len(tags[1]) == 7
        for example_tags in tags:
            for tag_id in example_tags:
                tag = self.model.vocab.get_token_from_index(tag_id, namespace="labels")
                assert tag in {'O', 'I-ORG', 'I-PER', 'I-LOC'}

    def test_file_archiving(self):
        # This happens to be a good place to test auxiliary file archiving.

        # Train the model
        params = Params.from_file('tests/fixtures/elmo/config/characters_token_embedder.json')
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

        # Which should properly contain { hocon_key -> original_filename }
        with open(fta_file) as fta:
            files_to_archive = json.loads(fta.read())

        assert files_to_archive == {
                'model.text_field_embedder.elmo.options_file': 'tests/fixtures/elmo/options.json',
                'model.text_field_embedder.elmo.weight_file': 'tests/fixtures/elmo/lm_weights.hdf5'
        }

        # Check that the unarchived contents of those files match the original contents.
        for key, original_filename in files_to_archive.items():
            new_filename = os.path.join(unarchive_dir, "fta", key)
            assert filecmp.cmp(original_filename, new_filename)
