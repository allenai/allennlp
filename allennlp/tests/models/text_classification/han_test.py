# pylint: disable=no-self-use,invalid-name
import numpy
from numpy.testing import assert_almost_equal
from models import han
from modules.seq2vec_encoders import han_attention
from allennlp.common.testing import ModelTestCase


class TestHAN(ModelTestCase):
    def setUp(self):
        super(TestHAN, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT + 'han/experiment.json',
                          self.FIXTURES_ROOT + 'data/ag.jsonl')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
