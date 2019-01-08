# pylint: disable=no-self-use,invalid-name
import numpy
from numpy.testing import assert_almost_equal
from models import seq2vec_classifier
from allennlp.common.testing import ModelTestCase


class TestSeq2VecClassifier(ModelTestCase):
    def setUp(self):
        super(TestSeq2VecClassifier, self).setUp()
        self.FIXTURES_ROOT = "/home/ubuntu/doc_classifiers/tests/fixtures/"
        self.set_up_model(self.FIXTURES_ROOT + 'seq2vec_classifier/experiment.json',
                    self.FIXTURES_ROOT + 'data/ag.jsonl')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
