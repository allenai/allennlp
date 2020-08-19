import logging
import os
import pathlib
import shutil
import tempfile
from unittest import mock

from allennlp.common.checks import log_pytorch_version_info

TEST_DIR = tempfile.mkdtemp(prefix="allennlp_tests")


class AllenNlpTestCase:
    """
    A custom subclass of `unittest.TestCase` that disables some of the more verbose AllenNLP
    logging and that creates and destroys a temp directory as a test fixture.
    """

    PROJECT_ROOT = (pathlib.Path(__file__).parent / ".." / ".." / "..").resolve()
    MODULE_ROOT = PROJECT_ROOT / "allennlp"
    TOOLS_ROOT = MODULE_ROOT / "tools"
    TESTS_ROOT = PROJECT_ROOT / "tests"
    FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"

    def setup_method(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.DEBUG
        )
        # Disabling some of the more verbose logging statements that typically aren't very helpful
        # in tests.
        logging.getLogger("allennlp.common.params").disabled = True
        logging.getLogger("allennlp.nn.initializers").disabled = True
        logging.getLogger("allennlp.modules.token_embedders.embedding").setLevel(logging.INFO)
        logging.getLogger("urllib3.connectionpool").disabled = True
        log_pytorch_version_info()

        self.TEST_DIR = pathlib.Path(TEST_DIR)

        os.makedirs(self.TEST_DIR, exist_ok=True)

        # Due to a bug in pytest we'll end up with a bunch of logging errors if we try to
        # log anything within an 'atexit' hook.
        # When https://github.com/pytest-dev/pytest/issues/5502 is fixed we should
        # be able to remove this work-around.
        def _cleanup_archive_dir_without_logging(path: str):
            if os.path.exists(path):
                shutil.rmtree(path)

        self.patcher = mock.patch(
            "allennlp.models.archival._cleanup_archive_dir", _cleanup_archive_dir_without_logging
        )
        self.mock_cleanup_archive_dir = self.patcher.start()

    def teardown_method(self):
        shutil.rmtree(self.TEST_DIR)
        self.patcher.stop()
