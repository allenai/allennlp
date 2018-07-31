# pylint: disable=invalid-name,no-self-use,protected-access
from collections import namedtuple
import os

from flaky import flaky
from numpy.testing import assert_almost_equal
import torch

from allennlp.common import Params
from allennlp.common.testing import ModelTestCase

class AtisSemanticParserTest(ModelTestCase):
    def setUp(self):
        super(AtisSemanticParserTest, self).setUp()
        self.set_up_model(str(self.FIXTURES_ROOT / "semantic_parsing" / "atis" / "experiment.json"),
                          str(self.FIXTURES_ROOT / "data" / "atis" / "sample.json"))

    def test_atis_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)




