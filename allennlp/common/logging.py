"""
A custom 'logging.Logger' that has additional {warning,debug,etc.}_once() methods
which allow for logged methods to be logged only once. Uses a 'logging.Filter'
to enforce the 'log_once' restriction.
"""
import logging

LOGGED_ONCE_MSG_PREFIX = "LOGGED ONCE: "


class DuplicateFilterByPrefix(logging.Filter):
    """
    A 'logging.Filter' subclass which filters duplicate messages if they have the LOGGED_ONCE_MSG_PREFIX.
    """

    def __init__(self):
        self.msgs = set()

    def filter(self, record):
        allow = True
        if record.msg.startswith(LOGGED_ONCE_MSG_PREFIX):
            allow = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return allow


class AllenNlpLogger(logging.Logger):
    """
    A custom subclass of 'logging.Logger' that uses a filter to implement {LOG_LEVEL}_once() methods.
    """

    def debug_once(self, msg, *args, **kwargs):
        msg = LOGGED_ONCE_MSG_PREFIX + msg
        self.debug(msg, *args, **kwargs)

    def info_once(self, msg, *args, **kwargs):
        msg = LOGGED_ONCE_MSG_PREFIX + msg
        self.info(msg, *args, **kwargs)

    def warning_once(self, msg, *args, **kwargs):
        msg = LOGGED_ONCE_MSG_PREFIX + msg
        self.warning(msg, *args, **kwargs)

    def error_once(self, msg, *args, **kwargs):
        msg = LOGGED_ONCE_MSG_PREFIX + msg
        self.error(msg, *args, **kwargs)

    def critical_once(self, msg, *args, **kwargs):
        msg = LOGGED_ONCE_MSG_PREFIX + msg
        self.critical(msg, *args, **kwargs)


logging.setLoggerClass(AllenNlpLogger)


def getLogger(name):
    """
    A method like 'logging.getLogger' but additionally adds the custom filter.
    """
    logger = logging.getLogger(name)
    filter = DuplicateFilterByPrefix()
    logger.addFilter(filter)
    return logger
