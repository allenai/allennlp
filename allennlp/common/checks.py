"""
Functions and exceptions for checking that
AllenNLP and its models are configured correctly.
"""

import logging
from typing import Union, List
from torch import cuda

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


def check_for_gpu(device_id: Union[int, List]):
    if device_id is None:
        device_ids = []
    elif isinstance(device_id, int):
        device_ids = [device_id]
    else:
        device_ids = device_id

    if any([device_id >= cuda.device_count() for device_id in device_ids]):
        raise ConfigurationError("Experiment specified a GPU but none is available;"
                                 " if you want to run on CPU use the override"
                                 " 'trainer.cuda_device=-1' in the json config file.")
