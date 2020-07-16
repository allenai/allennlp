import pytest

from allennlp.common.testing import ModelTestCase

_test_override_matrix = [
    '{"dataset_reader": {"mask_prepositions_verbs": %s, "drop_prepositions_verbs": %s}}'
    % (mpv, dpv)
    for mpv in ["true", "false"]
    for dpv in ["true", "false"]
]


class TestVilbert(ModelTestCase):
    @pytest.mark.parametrize("overrides", _test_override_matrix)
    def test_vilbert_can_train_save_and_load(self, overrides: str):
        param_file = self.FIXTURES_ROOT / "vilbert" / "experiment.json"
        self.ensure_model_can_train_save_and_load(param_file, overrides=overrides)
