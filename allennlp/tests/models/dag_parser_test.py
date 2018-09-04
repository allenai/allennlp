# pylint: disable=no-self-use,invalid-name,no-value-for-parameter

import torch

from allennlp.common.testing.model_test_case import ModelTestCase

class DagParserTest(ModelTestCase):

    def setUp(self):
        super(DagParserTest, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / "dag_parser" / "experiment.json",
                          self.FIXTURES_ROOT / "data" / "dm.sdp")

    def test_dag_parser_can_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()
