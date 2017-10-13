# pylint: disable=no-self-use,invalid-name

from flaky import flaky
from allennlp.common.testing import ModelTestCase


class CorefTest(ModelTestCase):
    def setUp(self):
        super(CorefTest, self).setUp()
        self.set_up_model('tests/fixtures/coref/experiment.json',
                          'tests/fixtures/data/coref/sample.gold_conll')

    def test_coref_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    @flaky
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()
