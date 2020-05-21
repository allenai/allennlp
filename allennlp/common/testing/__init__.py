"""
Utilities and helpers for writing tests.
"""
import torch
import pytest

from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.common.testing.model_test_case import ModelTestCase


_available_devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def multi_device(test_method):
    """
    Decorator that provides an argument `device` of type `str` for each available PyTorch device.
    """
    return pytest.mark.parametrize("device", _available_devices)(pytest.mark.gpu(test_method))


def requires_gpu(test_method):
    """
    Decorator to indicate that a test requires a GPU device.
    """
    return pytest.mark.gpu(
        pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device registered.")(
            test_method
        )
    )


def requires_multi_gpu(test_method):
    """
    Decorator to indicate that a test requires multiple GPU devices.
    """
    return pytest.mark.gpu(
        pytest.mark.skipif(torch.cuda.device_count() < 2, reason="2 or more GPUs required.")(
            test_method
        )
    )


def cpu_or_gpu(test_method):
    """
    Decorator to indicate that a test should run on both CPU and GPU
    """
    return pytest.mark.gpu(test_method)
