from allennlp.common.testing import ModelTestCase


class TestVilbert(ModelTestCase):
    def test_simple_tagger_can_train_save_and_load(self):
        param_file = self.FIXTURES_ROOT / "vilbert" / "experiment.json"
        self.ensure_model_can_train_save_and_load(param_file)
