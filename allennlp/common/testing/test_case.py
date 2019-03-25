# pylint: disable=invalid-name,protected-access
import logging
import os
import pathlib
import shutil
import tempfile
from unittest import TestCase

from allennlp.common.checks import log_pytorch_version_info

TEST_DIR = tempfile.mkdtemp(prefix="allennlp_tests")

class AllenNlpTestCase(TestCase):  # pylint: disable=too-many-public-methods
    """
    A custom subclass of :class:`~unittest.TestCase` that disables some of the
    more verbose AllenNLP logging and that creates and destroys a temp directory
    as a test fixture.
    """
    PROJECT_ROOT = (pathlib.Path(__file__).parent / ".." / ".." / "..").resolve()  # pylint: disable=no-member
    MODULE_ROOT = PROJECT_ROOT / "allennlp"
    TOOLS_ROOT = MODULE_ROOT / "tools"
    TESTS_ROOT = MODULE_ROOT / "tests"
    FIXTURES_ROOT = TESTS_ROOT / "fixtures"

    def setUp(self):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            level=logging.DEBUG)
        # Disabling some of the more verbose logging statements that typically aren't very helpful
        # in tests.
        logging.getLogger('allennlp.common.params').disabled = True
        logging.getLogger('allennlp.nn.initializers').disabled = True
        logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)
        logging.getLogger('urllib3.connectionpool').disabled = True
        log_pytorch_version_info()

        self.TEST_DIR = pathlib.Path(TEST_DIR)

        os.makedirs(self.TEST_DIR, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.TEST_DIR)
