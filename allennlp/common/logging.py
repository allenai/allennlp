import logging
from logging import Filter
import os
from os import PathLike
from typing import Union

import sys


class AllenNlpLogger(logging.Logger):
    """
    A custom subclass of 'logging.Logger' that keeps a set of messages to
    implement {debug,info,etc.}_once() methods.
    """

    def __init__(self, name):
        super().__init__(name)
        self._seen_msgs = set()

    def debug_once(self, msg, *args, **kwargs):
        if msg not in self._seen_msgs:
            self.debug(msg, *args, **kwargs)
            self._seen_msgs.add(msg)

    def info_once(self, msg, *args, **kwargs):
        if msg not in self._seen_msgs:
            self.info(msg, *args, **kwargs)
            self._seen_msgs.add(msg)

    def warning_once(self, msg, *args, **kwargs):
        if msg not in self._seen_msgs:
            self.warning(msg, *args, **kwargs)
            self._seen_msgs.add(msg)

    def error_once(self, msg, *args, **kwargs):
        if msg not in self._seen_msgs:
            self.error(msg, *args, **kwargs)
            self._seen_msgs.add(msg)

    def critical_once(self, msg, *args, **kwargs):
        if msg not in self._seen_msgs:
            self.critical(msg, *args, **kwargs)
            self._seen_msgs.add(msg)


logging.setLoggerClass(AllenNlpLogger)
logger = logging.getLogger(__name__)


FILE_FRIENDLY_LOGGING: bool = False
"""
If this flag is set to `True`, we add newlines to tqdm output, even on an interactive terminal, and we slow
down tqdm's output to only once every 10 seconds.

By default, it is set to `False`.
"""


class ErrorFilter(Filter):
    """
    Filters out everything that is at the ERROR level or higher. This is meant to be used
    with a stdout handler when a stderr handler is also configured. That way ERROR
    messages aren't duplicated.
    """

    def filter(self, record):
        return record.levelno < logging.ERROR


def prepare_global_logging(
    serialization_dir: Union[str, PathLike], rank: int = 0, world_size: int = 1,
) -> None:
    root_logger = logging.getLogger()

    # create handlers
    if world_size == 1:
        log_file = os.path.join(serialization_dir, "out.log")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    else:
        log_file = os.path.join(serialization_dir, f"out_worker{rank}.log")
        formatter = logging.Formatter(
            f"{rank} | %(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
    file_handler = logging.FileHandler(log_file)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stderr_handler = logging.StreamHandler(sys.stderr)

    handler: logging.Handler
    for handler in [file_handler, stdout_handler, stderr_handler]:
        handler.setFormatter(formatter)

    # Remove the already set handlers in root logger.
    # Not doing this will result in duplicate log messages
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

    if os.environ.get("ALLENNLP_DEBUG"):
        LEVEL = logging.DEBUG
    else:
        level_name = os.environ.get("ALLENNLP_LOG_LEVEL")
        LEVEL = logging._nameToLevel.get(level_name, logging.INFO)

    file_handler.setLevel(LEVEL)
    stdout_handler.setLevel(LEVEL)
    stdout_handler.addFilter(ErrorFilter())  # Make sure errors only go to stderr
    stderr_handler.setLevel(logging.ERROR)
    root_logger.setLevel(LEVEL)

    # put all the handlers on the root logger
    root_logger.addHandler(file_handler)
    if rank == 0:
        root_logger.addHandler(stdout_handler)
        root_logger.addHandler(stderr_handler)

    # write uncaught exceptions to the logs
    def excepthook(exctype, value, traceback):
        # For a KeyboardInterrupt, call the original exception handler.
        if issubclass(exctype, KeyboardInterrupt):
            sys.__excepthook__(exctype, value, traceback)
            return
        root_logger.critical("Uncaught exception", exc_info=(exctype, value, traceback))

    sys.excepthook = excepthook

    # also log tqdm
    from allennlp.common.tqdm import logger as tqdm_logger

    tqdm_logger.addHandler(file_handler)
