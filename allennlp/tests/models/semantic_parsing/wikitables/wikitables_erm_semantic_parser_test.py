# pylint: disable=no-self-use
import os
import pytest

from allennlp.common.testing import ModelTestCase
from allennlp.semparse.executors.wikitables_sempre_executor import SEMPRE_ABBREVIATIONS_PATH, SEMPRE_GRAMMAR_PATH

@pytest.mark.java
class WikiTablesErmSemanticParserTest(ModelTestCase):
    def setUp(self):
        self.should_remove_sempre_abbreviations = not os.path.exists(SEMPRE_ABBREVIATIONS_PATH)
        self.should_remove_sempre_grammar = not os.path.exists(SEMPRE_GRAMMAR_PATH)

        # The model tests are run with respect to the module root, so check if abbreviations
        # and grammar already exist there (since we want to clean up module root after test)
        self.module_root_abbreviations_path = self.MODULE_ROOT / "data" / "abbreviations.tsv"
        self.module_root_grammar_path = self.MODULE_ROOT / "data" / "grow.grammar"
        self.should_remove_root_sempre_abbreviations = not os.path.exists(self.module_root_abbreviations_path)
        self.should_remove_root_sempre_grammar = not os.path.exists(self.module_root_grammar_path)

        super(WikiTablesErmSemanticParserTest, self).setUp()
        self.set_up_model(str(self.FIXTURES_ROOT / "semantic_parsing" / "wikitables" / "experiment-erm.json"),
                          str(self.FIXTURES_ROOT / "data" / "wikitables" / "sample_data.examples"))

    def tearDown(self):
        super().tearDown()
        # We don't want to leave generated files around just from running tests...
        if self.should_remove_sempre_abbreviations and os.path.exists(SEMPRE_ABBREVIATIONS_PATH):
            os.remove(SEMPRE_ABBREVIATIONS_PATH)
        if self.should_remove_sempre_grammar and os.path.exists(SEMPRE_GRAMMAR_PATH):
            os.remove(SEMPRE_GRAMMAR_PATH)
        if self.should_remove_root_sempre_abbreviations and os.path.exists(self.module_root_abbreviations_path):
            os.remove(self.module_root_abbreviations_path)
        if self.should_remove_root_sempre_grammar and os.path.exists(self.module_root_grammar_path):
            os.remove(self.module_root_grammar_path)

    def test_model_can_train_save_and_load(self):
        # We have very few embedded actions on our agenda, and so it's rare that this parameter
        # actually gets used.  We know this parameter works from our NLVR ERM test, so it's easier
        # to just ignore it here than to try to finagle the test to make it so this has a non-zero
        # gradient.
        ignore = {'_decoder_step._checklist_multiplier'}
        self.ensure_model_can_train_save_and_load(self.param_file, gradients_to_ignore=ignore)
