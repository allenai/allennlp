"""
Functions and exceptions for checking that
AllenNLP and its models are configured correctly.
"""
from typing import Union, List

import logging
import re
import subprocess

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


class ExperimentalFeatureWarning(RuntimeWarning):
    """
    A warning that you are using an experimental feature
    that may change or be deleted.
    """
    pass


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

def parse_cuda_device(cuda_device: Union[str, int, List[int]]) -> Union[int, List[int]]:
    """
    Disambiguates single GPU and multiple GPU settings for cuda_device param.
    """
    def from_list(strings):
        if len(strings) > 1:
            return [int(d) for d in strings]
        elif len(strings) == 1:
            return int(strings[0])
        else:
            return -1

    if isinstance(cuda_device, str):
        return from_list(re.split(r',\s*', cuda_device))
    elif isinstance(cuda_device, int):
        return cuda_device
    elif isinstance(cuda_device, list):
        return from_list(cuda_device)
    else:
        # TODO(brendanr): Determine why mypy can't tell that this matches the Union.
        return int(cuda_device)  # type: ignore

def check_for_gpu(device_id: Union[int, list]):
    device_id = parse_cuda_device(device_id)
    if isinstance(device_id, list):
        for did in device_id:
            check_for_gpu(did)
    elif device_id is not None and device_id >= 0:
        num_devices_available = cuda.device_count()
        if num_devices_available == 0:
            raise ConfigurationError("Experiment specified a GPU but none is available;"
                                     " if you want to run on CPU use the override"
                                     " 'trainer.cuda_device=-1' in the json config file.")
        elif device_id >= num_devices_available:
            raise ConfigurationError(f"Experiment specified GPU device {device_id}"
                                     f" but there are only {num_devices_available} devices "
                                     f" available.")


def check_for_java() -> bool:
    try:
        java_version = subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT)
        return 'version' in java_version.decode()
    except FileNotFoundError:
        return False
