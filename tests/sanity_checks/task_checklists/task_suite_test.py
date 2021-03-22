import pytest
from allennlp.sanity_checks.task_checklists.task_suite import TaskSuite
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common.testing.checklist_test import FakeTaskSuite  # noqa: F401


class TestTaskSuite(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        archive = load_archive(
            self.FIXTURES_ROOT / "basic_classifier" / "serialization" / "model.tar.gz"
        )
        self.predictor = Predictor.from_archive(archive)

    def test_load_from_suite_file(self):
        suite_file = str(self.FIXTURES_ROOT / "task_suites" / "fake_suite.tar.gz")

        task_suite = TaskSuite.constructor(suite_file=suite_file)

        assert len(task_suite.suite.tests) == 1

    def test_load_by_name(self):

        task_suite = TaskSuite.constructor(name="fake-task-suite")

        assert task_suite._fake_arg1 is None
        assert task_suite._fake_arg2 is None

        assert len(task_suite.suite.tests) == 1

        with pytest.raises(ConfigurationError):
            TaskSuite.constructor(name="suite-that-does-not-exist")

    def test_load_with_extra_args(self):
        extra_args = {"fake_arg1": "some label"}
        task_suite = TaskSuite.constructor(name="fake-task-suite", extra_args=extra_args)
        assert task_suite._fake_arg1 == "some label"

    def test_prediction_and_confidence_scores_function_needs_implementation(self):

        task_suite = TaskSuite.constructor(name="fake-task-suite")

        with pytest.raises(NotImplementedError):
            task_suite.run(self.predictor)
