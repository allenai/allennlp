# pylint: disable=invalid-name
import numpy
import torch

from allennlp.common.testing import ModelTestCase


class Event2MindTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / "event2mind" / "experiment.json",
                          self.FIXTURES_ROOT / "data" / "event2mind_medium.csv")

    def test_encoder_decoder_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
