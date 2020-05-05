import logging
from logging import Filter
import os
import sys
from typing import Optional

from allennlp.common.tee import TeeHandler
from allennlp.common.tqdm import Tqdm


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


class ErrorFilter(Filter):
    """
    Filters out everything that is at the ERROR level or higher. This is meant to be used
    with a stdout handler when a stderr handler is also configured. That way ERROR
    messages aren't duplicated.
    """

    def filter(self, record):
        return record.levelno < logging.ERROR


class WorkerLogFilter(Filter):
    def __init__(self, rank=-1):
        super().__init__()
        self._rank = rank

    def filter(self, record):
        if self._rank != -1:
            record.msg = f"Rank {self._rank} | {record.msg}"
        return True


def prepare_global_logging(
    serialization_dir: str, file_friendly_logging: bool, rank: int = 0, world_size: int = 1
) -> None:
    # If we don't have a terminal as stdout,
    # force tqdm to be nicer.
    if not sys.stdout.isatty():
        file_friendly_logging = True

    Tqdm.set_slower_interval(file_friendly_logging)

    stdout_file: str
    stderr_file: str
    worker_filter: Optional[WorkerLogFilter] = None
    if world_size == 1:
        # This case is not distributed training and hence will stick to the older
        # log file names
        stdout_file = os.path.join(serialization_dir, "stdout.log")
        stderr_file = os.path.join(serialization_dir, "stderr.log")
    else:
        # Create log files with worker ids
        stdout_file = os.path.join(serialization_dir, f"stdout_worker{rank}.log")
        stderr_file = os.path.join(serialization_dir, f"stderr_worker{rank}.log")

        # This adds the worker's rank to messages being logged to files.
        # This will help when combining multiple worker log files using `less` command.
        worker_filter = WorkerLogFilter(rank)

    # Patch stdout/err.
    stdout_patch = TeeHandler(
        stdout_file,
        sys.stdout,
        file_friendly_terminal_output=file_friendly_logging,
        silent=rank != 0,  # don't print to terminal from non-master workers.
    )
    sys.stdout = stdout_patch  # type: ignore
    stderr_patch = TeeHandler(
        stderr_file,
        sys.stderr,
        file_friendly_terminal_output=file_friendly_logging,
        silent=rank != 0,  # don't print to terminal from non-master workers.
    )
    sys.stderr = stderr_patch  # type: ignore

    # Handlers for stdout/err logging
    output_handler = logging.StreamHandler(sys.stdout)
    error_handler = logging.StreamHandler(sys.stderr)

    if worker_filter is not None:
        output_handler.addFilter(worker_filter)
        error_handler.addFilter(worker_filter)

    root_logger = logging.getLogger()

    # Remove the already set stream handler in root logger.
    # Not doing this will result in duplicate log messages
    # printed in the console
    if len(root_logger.handlers) > 0:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    output_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)

    if os.environ.get("ALLENNLP_DEBUG"):
        LEVEL = logging.DEBUG
    else:
        level_name = os.environ.get("ALLENNLP_LOG_LEVEL")
        LEVEL = logging._nameToLevel.get(level_name, logging.INFO)

    output_handler.setLevel(LEVEL)
    error_handler.setLevel(logging.ERROR)

    # filter out everything at the ERROR or higher level for output stream
    # so that error messages don't appear twice in the logs.
    output_handler.addFilter(ErrorFilter())

    root_logger.addHandler(output_handler)
    root_logger.addHandler(error_handler)

    root_logger.setLevel(LEVEL)
