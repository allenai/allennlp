# pylint: disable=invalid-name,protected-access
import logging
import os
import pathlib
import shutil
import tempfile
from unittest import TestCase

from allennlp.common.checks import log_pytorch_version_info


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

        self.TEST_DIR = pathlib.Path(tempfile.mkdtemp())
        print(f"\nZZZ CREATED TEST_DIR {self.TEST_DIR}")

    def tearDown(self):
        import psutil
        print(f"ZZZ TEARING DOWN {self.__class__.__name__} {psutil.Process()}")
        print("ZZZ OPEN FILES: " + str(sum([len(p.open_files()) for p in psutil.Process().children()])))
        for child in psutil.Process().children():
            print(f"ZZZ {child}")
            print(f"ZZZ  " + ", ".join([x.path for x in child.open_files()]))
        try:
            shutil.rmtree(self.TEST_DIR)
        except Exception as e:
            print("ZZZ {e.__class__.__name__}")
            import traceback
            import sys
            traceback.print_exc(file=sys.stdout)
        if self.TEST_DIR.exists():
            print(f"\nZZZ DELETED {self.TEST_DIR} but it still exists ({self.__class__.__name__}).")
        else:
            print(f"\nZZZ DELETED {self.TEST_DIR} successfully. ({self.__class__.__name__})")
