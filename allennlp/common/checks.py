"""
Functions and exceptions for checking that
AllenNLP and its models are configured correctly.
"""

import logging
import os

REQUIRED_PYTHONHASHSEED = '2157'

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ConfigurationError(Exception):
    """
    The exception raised by any AllenNLP object when it's misconfigured
    (e.g. missing properties, invalid properties, unknown properties).
    """
    def __init__(self, message):
        super(ConfigurationError, self).__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)


def ensure_pythonhashseed_set():
    """
    Makes sure that the ``PYTHONHASHSEED`` environment variable is set to the correct value,
    which it must be in order to get repeatable results (and for the tests to pass).
    """

    message = """You must set PYTHONHASHSEED to %s so we get repeatable results and tests pass.
    You can do this with the command `export PYTHONHASHSEED=%s`.
    See https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED for more info.
    """
    assert os.environ.get('PYTHONHASHSEED', None) == REQUIRED_PYTHONHASHSEED, \
        message % (REQUIRED_PYTHONHASHSEED, REQUIRED_PYTHONHASHSEED)


def log_pytorch_version_info():
    import torch
    logger.info("Pytorch version: " + torch.__version__)
