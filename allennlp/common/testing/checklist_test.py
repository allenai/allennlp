from typing import Optional
from checklist.test_suite import TestSuite
from checklist.test_types import MFT as MinimumFunctionalityTest
from allennlp.confidence_checks.task_checklists.task_suite import TaskSuite


@TaskSuite.register("fake-task-suite")
class FakeTaskSuite(TaskSuite):
    """
    Fake checklist suite for testing purpose.
    """

    def __init__(
        self,
        suite: Optional[TestSuite] = None,
        fake_arg1: Optional[int] = None,
        fake_arg2: Optional[int] = None,
    ):
        self._fake_arg1 = fake_arg1
        self._fake_arg2 = fake_arg2

        if not suite:
            suite = TestSuite()

        # Adding a simple checklist test.
        test = MinimumFunctionalityTest(
            ["sentence 1", "sentence 2"],
            labels=0,
            name="fake test 1",
            capability="fake capability",
            description="Test's description",
        )
        suite.add(test)

        super().__init__(suite)
