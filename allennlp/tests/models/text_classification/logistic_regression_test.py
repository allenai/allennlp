# pylint: disable=no-self-use,invalid-name
import numpy
from numpy.testing import assert_almost_equal
from models import logistic_regression
from allennlp.common.testing import ModelTestCase


class TestLogisticRegression(ModelTestCase):
    def setUp(self):
        super(TestLogisticRegression, self).setUp()
        self.FIXTURES_ROOT = "/home/ubuntu/doc_classifiers/tests/fixtures/"
        self.set_up_model(self.FIXTURES_ROOT + 'logistic_regression/experiment.json',
                          self.FIXTURES_ROOT + 'data/ag.jsonl')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
