import json
import pytest

from allennlp.common.testing import ModelTestCase


class ModelWithIncorrectValidationMetricTest(ModelTestCase):
    """
    This test case checks some validating functionality that is implemented
    in `ensure_model_can_train_save_and_load`
    """

    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            self.FIXTURES_ROOT / "simple_tagger" / "model_test_case.jsonnet",
            self.FIXTURES_ROOT / "data" / "sequence_tagging.tsv",
        )

    def test_01_test_validation_metric_does_not_exist(self):
        overrides = {"trainer.num_epochs": 2}
        pytest.raises(
            AssertionError,
            self.ensure_model_can_train_save_and_load,
            self.param_file,
            metric_to_check="non_existent_metric",
            metric_terminal_value=0.0,
            overrides=json.dumps(overrides),
        )

    def test_02a_test_validation_metric_terminal_value_not_set(self):
        pytest.raises(
            AssertionError,
            self.ensure_model_can_train_save_and_load,
            self.param_file,
            metric_to_check="accuracy",
            metric_terminal_value=None,
        )

    def test_02b_test_validation_metric_terminal_value_not_met(self):
        pytest.raises(
            AssertionError,
            self.ensure_model_can_train_save_and_load,
            self.param_file,
            metric_to_check="accuracy",
            metric_terminal_value=0.0,
        )

    def test_03_test_validation_metric_exists_and_its_terminal_value_is_met(self):
        self.ensure_model_can_train_save_and_load(
            self.param_file,
            metric_to_check="accuracy",
            metric_terminal_value=1.0,
        )
