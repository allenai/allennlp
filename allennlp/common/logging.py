"""
A custom 'logging.Logger' that has additional {warning,debug,etc.}_once() methods
which allow for logged methods to be logged only once.
"""
import logging


class AllenNlpLogger(logging.Logger):
    """
    A custom subclass of 'logging.Logger' that keeps a set of messages to
    implement {debug,info,etc.}_once() methods.
    """

    def __init__(self, name):
        super().__init__(name)
        self.msgs = set()

    def debug_once(self, msg, *args, **kwargs):
        if msg not in self.msgs:
            self.debug(msg, *args, **kwargs)
        self.msgs.add(msg)

    def info_once(self, msg, *args, **kwargs):
        if msg not in self.msgs:
            self.info(msg, *args, **kwargs)
        self.msgs.add(msg)

    def warning_once(self, msg, *args, **kwargs):
        if msg not in self.msgs:
            self.warning(msg, *args, **kwargs)
        self.msgs.add(msg)

    def error_once(self, msg, *args, **kwargs):
        if msg not in self.msgs:
            self.error(msg, *args, **kwargs)
        self.msgs.add(msg)

    def critical_once(self, msg, *args, **kwargs):
        if msg not in self.msgs:
            self.critical(msg, *args, **kwargs)
        self.msgs.add(msg)


logging.setLoggerClass(AllenNlpLogger)
