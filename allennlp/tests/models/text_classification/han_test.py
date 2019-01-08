# pylint: disable=invalid-name,no-self-use,protected-access
import numpy
from numpy.testing import assert_almost_equal
from allennlp.models import HierarchicalAttentionNetwork
from allennlp.modules.seq2vec_encoders import AttentionEncoder
from allennlp.common.testing import ModelTestCase


class TestHAN(ModelTestCase):
    def setUp(self):
        super(TestHAN, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'han' / 'experiment.json',
                          self.FIXTURES_ROOT / 'data' / 'textcat' / 'ag.jsonl')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
