import pytest

from allennlp.common.testing import ModelTestCase


class TestVilbert(ModelTestCase):
    @pytest.mark.parametrize(
        "overrides",
        [
            '{"dataset_reader": {"mask_prepositions_verbs": %s, "drop_prepositions_verbs": %s}}'
            % (mpv, dpv)
            for mpv in ["true", "false"]
            for dpv in ["true", "false"]
        ],
    )
    def test_model_can_train_save_and_load(self, overrides: str):
        param_file = self.FIXTURES_ROOT / "vilbert" / "experiment.json"
        self.ensure_model_can_train_save_and_load(param_file, overrides=overrides)

    def test_model_can_train_save_and_load_from_huggingface(self):
        param_file = self.FIXTURES_ROOT / "vilbert" / "experiment_from_huggingface.jsonnet"
        self.ensure_model_can_train_save_and_load(param_file)
