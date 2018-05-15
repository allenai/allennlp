# pylint: disable=no-self-use,invalid-name,protected-access
import os
import shutil

import torch
from numpy.testing import assert_almost_equal

from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch
from allennlp.data.dataset_readers import WikiTablesDatasetReader
from allennlp.models.archival import load_archive
from allennlp.training.metrics.wikitables_accuracy import SEMPRE_DIR

class WikiTablesErmSemanticParserTest(ModelTestCase):
    def setUp(self):
        self.should_remove_sempre_dir = not os.path.exists(SEMPRE_DIR)
        super(WikiTablesErmSemanticParserTest, self).setUp()
        self.set_up_model(f"tests/fixtures/semantic_parsing/wikitables/experiment-erm.json",
                          "tests/fixtures/data/wikitables/sample_data.examples")

    def tearDown(self):
        super().tearDown()
        # We don't want to leave generated files around just from running tests...
        if self.should_remove_sempre_dir and os.path.exists(SEMPRE_DIR):
            shutil.rmtree('data')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_initialize_weights_from_archive(self):
        original_model_parameters = self.model.named_parameters()
        original_model_weights = {name: parameter.data.clone().numpy()
                                  for name, parameter in original_model_parameters}
        # pylint: disable=line-too-long
        mml_model_archive_file = "tests/fixtures/semantic_parsing/wikitables/serialization/model.tar.gz"
        archive = load_archive(mml_model_archive_file)
        archived_model_parameters = archive.model.named_parameters()
        self.model._initialize_weights_from_archive(archive)
        changed_model_parameters = dict(self.model.named_parameters())
        for name, archived_parameter in archived_model_parameters:
            archived_weight = archived_parameter.data.numpy()
            original_weight = original_model_weights[name]
            changed_weight = changed_model_parameters[name].data.numpy()
            # We want to make sure that the weights in the original model have indeed been changed
            # after a call to ``_initialize_weights_from_archive``.
            with self.assertRaises(AssertionError, msg=f"{name} has not changed"):
                assert_almost_equal(original_weight, changed_weight)
            # This also includes the sentence token embedder. Those weights will be the same
            # because the two models have the same vocabulary.
            assert_almost_equal(archived_weight, changed_weight)

    def test_untrained_outputs_are_same_as_initial_models(self):
        # We want to make sure that the outputs of an untrained mml initialized model are the
        # same as those from the original mml model.
        archive_file = 'tests/fixtures/semantic_parsing/wikitables/serialization/model.tar.gz'
        mml_archive = load_archive(archive_file)
        mml_model = mml_archive.model
        tables_directory = "tests/fixtures/data/wikitables/"
        dpd_directory = "tests/fixtures/data/wikitables/dpd_output/"
        examples_file = "tests/fixtures/data/wikitables/sample_data.examples"
        dataset_reader = WikiTablesDatasetReader(tables_directory=tables_directory,
                                                 dpd_output_directory=dpd_directory)
        dataset = Batch(dataset_reader.read(examples_file))
        dataset.index_instances(mml_model.vocab)
        mml_data = dataset.as_tensor_dict(cuda_device=-1, for_training=False)
        mml_model.training = False
        mml_outputs = mml_model(**mml_data)
        mml_logical_forms = mml_outputs["logical_form"]
        erm_data = self.dataset.as_tensor_dict(cuda_device=-1, for_training=False)
        self.model._initialize_weights_from_archive(mml_archive)
        # Overwriting the checklist multipliers to not let the checklist affect the action
        # predictions.
        model_parameters = dict(self.model.named_parameters())
        model_parameters["_decoder_step._unlinked_checklist_multiplier"].data.copy_(torch.FloatTensor([0.0]))
        model_parameters["_decoder_step._linked_checklist_multiplier"].data.copy_(torch.FloatTensor([0.0]))
        self.model.training = False
        erm_outputs = self.model(**erm_data)
        erm_logical_forms = erm_outputs["logical_form"]
        assert mml_logical_forms == erm_logical_forms
