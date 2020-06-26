"""
`allennlp.common.tqdm.Tqdm` wraps tqdm so we can add configurable
global defaults for certain tqdm parameters.
"""
import logging
from time import time

import sys

from allennlp.common import tee

try:
    SHELL = str(type(get_ipython()))  # type:ignore # noqa: F821
except:  # noqa: E722
    SHELL = ""

if "zmqshell.ZMQInteractiveShell" in SHELL:
    from tqdm import tqdm_notebook as _tqdm
else:
    from tqdm import tqdm as _tqdm

# This is necessary to stop tqdm from hanging
# when exceptions are raised inside iterators.
# It should have been fixed in 4.2.1, but it still
# occurs.
# TODO(Mark): Remove this once tqdm cleans up after itself properly.
# https://github.com/tqdm/tqdm/issues/469
_tqdm.monitor_interval = 0


logger = logging.getLogger("tqdm")
logger.propagate = False


class TqdmToLogsWriter(object):
    def __init__(self):
        self.last_message_written_time = 0

    def write(self, message):
        sys.stderr.write(message)
        now = time()
        if now - self.last_message_written_time >= 10 or "100%" in message:
            message = tee.replace_cr_with_newline(message)
            for message in message.split("\n"):
                message = message.strip()
                if len(message) > 0:
                    logger.info(message)
                    self.last_message_written_time = now

    def flush(self):
        sys.stderr.flush()


class Tqdm:
    @staticmethod
    def tqdm(*args, **kwargs):
        new_kwargs = {"file": TqdmToLogsWriter(), **kwargs}

        return _tqdm(*args, **new_kwargs)
