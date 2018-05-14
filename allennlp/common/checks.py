"""
Functions and exceptions for checking that
AllenNLP and its models are configured correctly.
"""

import logging

import torch

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


def log_pytorch_version_info():
    import torch
    logger.info("Pytorch version: %s", torch.__version__)


def check_dimensions_match(dimension_1: int,
                           dimension_2: int,
                           dim_1_name: str,
                           dim_2_name: str) -> None:
    if dimension_1 != dimension_2:
        raise ConfigurationError(f"{dim_1_name} must match {dim_2_name}, but got {dimension_1} "
                                 f"and {dimension_2} instead")


def cuda_is_valid(id: int):
    if id is None:
        # this branch is to ensure that existing tests which don't specify cuda pass.
        return
    try:
        with torch.cuda.device(id):
            pass
    except Exception:
        raise ConfigurationError("Experiment specified a GPU but none is available;"
                                 " if you want to run on CPU use the following override trainer.cuda_device=-1 in the json config file")


