# pylint: disable=no-self-use
import os
import shutil

from allennlp.common.testing import ModelTestCase
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
