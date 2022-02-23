import logging
import os
import pathlib
import shutil
import tempfile

from allennlp.common.checks import log_pytorch_version_info

TEST_DIR = tempfile.mkdtemp(prefix="allennlp_tests")


class AllenNlpTestCase:
    """
    A custom testing class that disables some of the more verbose AllenNLP
    logging and that creates and destroys a temp directory as a test fixture.
    """

    PROJECT_ROOT = (pathlib.Path(__file__).parent / ".." / ".." / "..").resolve()
    MODULE_ROOT = PROJECT_ROOT / "allennlp"
    TOOLS_ROOT = MODULE_ROOT / "tools"
    # to run test suite with finished package, which does not contain
    # tests & fixtures, we must be able to look them up somewhere else
    PROJECT_ROOT_FALLBACK = (
        # users wanting to run test suite for installed package
        pathlib.Path(os.environ["ALLENNLP_SRC_DIR"])
        if "ALLENNLP_SRC_DIR" in os.environ
        else (
            # fallback for conda packaging
            pathlib.Path(os.environ["SRC_DIR"])
            if "CONDA_BUILD" in os.environ
            # stay in-tree
            else PROJECT_ROOT
        )
    )
    TESTS_ROOT = PROJECT_ROOT_FALLBACK / "tests"
    FIXTURES_ROOT = PROJECT_ROOT_FALLBACK / "test_fixtures"

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

    def teardown_method(self):
        shutil.rmtree(self.TEST_DIR)
