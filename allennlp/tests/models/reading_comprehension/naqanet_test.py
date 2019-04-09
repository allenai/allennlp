#pylint: disable=unused-import
from flaky import flaky

from allennlp.common.testing import ModelTestCase


class NumericallyAugmentedQaNetTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        print(self.FIXTURES_ROOT)
        self.set_up_model(self.FIXTURES_ROOT / "naqanet" / "experiment.json",
                          self.FIXTURES_ROOT / "data" / "drop.json")

    @flaky
    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
