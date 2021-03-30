import pytest
import sqlite3
from unittest.mock import call, Mock

from allennlp.common.testing import AllenNlpTestCase

from scripts.ai2_internal.resume_daemon import (
    BeakerStatus,
    create_table,
    handler,
    logger,
    resume,
    start_autoresume,
)

# Don't spam the log in tests.
logger.removeHandler(handler)


class ResumeDaemonTest(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.connection = sqlite3.connect(":memory:")
        create_table(self.connection)

    def test_create_beaker_status_works(self):
        status = BeakerStatus("stopped")
        assert status.name == "stopped"

    def test_create_beaker_status_throws(self):
        with pytest.raises(ValueError):
            status = BeakerStatus("garbage")
            assert status.name == "garbage"

    def test_does_nothing_on_empty_db(self):
        beaker = Mock()
        resume(self.connection, beaker)
        assert not beaker.method_calls

    def test_does_not_resume_a_running_experiment(self):
        beaker = Mock()
        experiment_id = "foo"
        start_autoresume(self.connection, experiment_id, 5)
        beaker.get_status.return_value = BeakerStatus.running
        resume(self.connection, beaker)
        beaker.get_status.assert_called()
        assert len(beaker.method_calls) == 1

    def test_does_not_resume_a_finished_experiment(self):
        beaker = Mock()
        experiment_id = "foo"
        start_autoresume(self.connection, experiment_id, 5)
        beaker.get_status.return_value = BeakerStatus.succeeded
        resume(self.connection, beaker)
        beaker.get_status.assert_called()
        assert len(beaker.method_calls) == 1

    def test_does_resume_a_preempted_experiment(self):
        beaker = Mock()
        experiment_id = "foo"
        start_autoresume(self.connection, experiment_id, 5)
        beaker.get_status.return_value = BeakerStatus.preempted
        beaker.resume.return_value = "foo2"
        resume(self.connection, beaker)
        beaker.get_status.assert_called()
        beaker.resume.assert_called()
        assert len(beaker.method_calls) == 2

    def test_respects_upper_bound_on_resumes(self):
        beaker = Mock()
        experiment_id = "foo"
        start_autoresume(self.connection, experiment_id, 5)
        beaker.get_status.return_value = BeakerStatus.preempted
        for i in range(10):
            beaker.resume.return_value = f"foo{i}"
            resume(self.connection, beaker)
        calls = [
            call.get_status("foo"),
            call.resume("foo"),
            call.get_status("foo0"),
            call.resume("foo0"),
            call.get_status("foo1"),
            call.resume("foo1"),
            call.get_status("foo2"),
            call.resume("foo2"),
            call.get_status("foo3"),
            call.resume("foo3"),
            call.get_status("foo4"),
        ]
        beaker.assert_has_calls(calls)

    def test_handles_a_realistic_scenario(self):
        beaker = Mock()
        experiment_id = "foo"
        start_autoresume(self.connection, experiment_id, 5)
        beaker.get_status.return_value = BeakerStatus.preempted
        for i in range(10):
            beaker.resume.return_value = f"foo{i}"
            if i == 2:
                beaker.get_status.return_value = BeakerStatus.succeeded
            resume(self.connection, beaker)
        calls = [
            call.get_status("foo"),
            call.resume("foo"),
            call.get_status("foo0"),
            call.resume("foo0"),
            call.get_status("foo1"),
        ]
        beaker.assert_has_calls(calls)
