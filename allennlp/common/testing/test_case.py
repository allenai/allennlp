import logging
import os
import pathlib
import shutil
import pytest
import tempfile

from allennlp.common.checks import log_pytorch_version_info


# TEST_DIR = tempfile.mkdtemp(prefix="allennlp_tests")


class AllenNlpTestCase:
    """
    A custom testing class that disables some of the more verbose AllenNLP
    logging and that creates and destroys a temp directory as a test fixture.
    """

    PROJECT_ROOT = (pathlib.Path(__file__).parent / ".." / ".." / "..").resolve()
    MODULE_ROOT = PROJECT_ROOT / "allennlp"
    TOOLS_ROOT = MODULE_ROOT / "tools"
    TESTS_ROOT = PROJECT_ROOT / "tests"
    FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"

    # Had issues with the old tempdir, so used this pytest fixture that is
    # always ran and will always create a tmpdir.
    @pytest.fixture(autouse=True)
    def test_dir(self, tmpdir):
        self.TEST_DIR = pathlib.Path(tmpdir.mkdir('semparse'))

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

    def teardown_method(self):
        shutil.rmtree(self.TEST_DIR)
