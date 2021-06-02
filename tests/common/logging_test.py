import os
import logging
import random

from allennlp.common.logging import AllenNlpLogger, prepare_global_logging
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.tqdm import Tqdm


class TestLogging(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        logger = logging.getLogger(str(random.random()))
        self.test_log_file = os.path.join(self.TEST_DIR, "test.log")
        logger.addHandler(logging.FileHandler(self.test_log_file))
        logger.setLevel(logging.DEBUG)
        self.logger = logger
        self._msg = "test message"

    def test_debug_once(self):
        self.logger.debug_once(self._msg)
        self.logger.debug_once(self._msg)

        with open(self.test_log_file, "r") as f:
            assert len(f.readlines()) == 1

    def test_info_once(self):
        self.logger.info_once(self._msg)
        self.logger.info_once(self._msg)

        with open(self.test_log_file, "r") as f:
            assert len(f.readlines()) == 1

    def test_warning_once(self):
        self.logger.warning_once(self._msg)
        self.logger.warning_once(self._msg)

        with open(self.test_log_file, "r") as f:
            assert len(f.readlines()) == 1

    def test_error_once(self):
        self.logger.error_once(self._msg)
        self.logger.error_once(self._msg)

        with open(self.test_log_file, "r") as f:
            assert len(f.readlines()) == 1

    def test_critical_once(self):
        self.logger.critical_once(self._msg)
        self.logger.critical_once(self._msg)

        with open(self.test_log_file, "r") as f:
            assert len(f.readlines()) == 1

    def test_debug_once_different_args(self):
        self.logger.debug_once("There are %d lights.", 4)
        self.logger.debug_once("There are %d lights.", 5)

        with open(self.test_log_file, "r") as f:
            assert len(f.readlines()) == 1

        assert len(self.logger._seen_msgs) == 1

    def test_getLogger(self):
        logger = logging.getLogger("test_logger")

        assert isinstance(logger, AllenNlpLogger)

    def test_reset_tqdm_logger_handlers(self):
        serialization_dir_a = os.path.join(self.TEST_DIR, "test_a")
        os.makedirs(serialization_dir_a, exist_ok=True)
        prepare_global_logging(serialization_dir_a)
        serialization_dir_b = os.path.join(self.TEST_DIR, "test_b")
        os.makedirs(serialization_dir_b, exist_ok=True)
        prepare_global_logging(serialization_dir_b)
        # Use range(1) to make sure there should be only 2 lines in the file (0% and 100%)
        for _ in Tqdm.tqdm(range(1)):
            pass
        with open(os.path.join(serialization_dir_a, "out.log"), "r") as f:
            assert len(f.readlines()) == 0
        with open(os.path.join(serialization_dir_b, "out.log"), "r") as f:
            assert len(f.readlines()) == 2
