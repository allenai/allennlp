# pylint: disable=invalid-name,protected-access
import logging
import os
import shutil
from unittest import TestCase

from allennlp.common.checks import log_pytorch_version_info


class AllenNlpTestCase(TestCase):  # pylint: disable=too-many-public-methods
    TEST_DIR = './TMP_TEST/'

    def setUp(self):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            level=logging.DEBUG)
        # Disabling some of the more verbose logging statements that typically aren't very helpful
        # in tests.
        logging.getLogger('allennlp.common.params').disabled = True
        logging.getLogger('allennlp.nn.initializers').disabled = True
        logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)
        log_pytorch_version_info()
        os.makedirs(self.TEST_DIR, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.TEST_DIR)
