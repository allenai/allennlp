import os
import logging
from allennlp.common.logging import (
    LOGGED_ONCE_MSG_PREFIX,
    DuplicateFilterByPrefix,
    AllenNlpLogger,
    getLogger,
)
from allennlp.common.testing import AllenNlpTestCase


class TestLogging(AllenNlpTestCase):
    i = 0

    def setUp(self):
        super().setUp()
        logger = getLogger(str(self.__class__.i))
        self.test_log_file = os.path.join(self.TEST_DIR, "test.log")
        hdlr = logging.FileHandler(self.test_log_file)
        logger.addHandler(hdlr)
        self.logger = logger
        self.msg = "test message"
        self.__class__.i += 1

    def testDuplicateFilterByPrefix(self):
        filter = DuplicateFilterByPrefix()

        # Filters duplicate message with prefix
        msg1 = logging.makeLogRecord({"msg": LOGGED_ONCE_MSG_PREFIX + self.msg})
        self.assertTrue(filter.filter(msg1))
        self.assertFalse(filter.filter(msg1))

        # Does not filter duplicate message without prefix
        msg2 = logging.makeLogRecord({"msg": self.msg})
        self.assertTrue(filter.filter(msg2))
        self.assertTrue(filter.filter(msg2))

    def test_debug_once(self):
        self.logger.debug_once(self.msg)
        self.logger.debug_once(self.msg)

        with open(self.test_log_file, "r") as f:
            assert len(f.readlines()) == 1

    def test_info_once(self):
        self.logger.info_once(self.msg)
        self.logger.info_once(self.msg)

        with open(self.test_log_file, "r") as f:
            assert len(f.readlines()) == 1

    def test_warning_once(self):
        self.logger.warning_once(self.msg)
        self.logger.warning_once(self.msg)

        with open(self.test_log_file, "r") as f:
            assert len(f.readlines()) == 1

    def test_error_once(self):
        self.logger.error_once(self.msg)
        self.logger.error_once(self.msg)

        with open(self.test_log_file, "r") as f:
            assert len(f.readlines()) == 1

    def test_critical_once(self):
        self.logger.critical_once(self.msg)
        self.logger.critical_once(self.msg)

        with open(self.test_log_file, "r") as f:
            assert len(f.readlines()) == 1

    def test_getLogger(self):
        logger = getLogger("test_logger")

        assert isinstance(logger, AllenNlpLogger)
        assert isinstance(logger.filters[0], DuplicateFilterByPrefix)
