import logging
import os
import pathlib
import shutil
import tempfile
from typing import Any, Iterable
from unittest import TestCase

import torch

from allennlp.common.checks import log_pytorch_version_info

TEST_DIR = tempfile.mkdtemp(prefix="allennlp_tests")


class AllenNlpTestCase(TestCase):
    """
    A custom subclass of `unittest.TestCase` that disables some of the more verbose AllenNLP
    logging and that creates and destroys a temp directory as a test fixture.
    """

    PROJECT_ROOT = (pathlib.Path(__file__).parent / ".." / ".." / "..").resolve()
    MODULE_ROOT = PROJECT_ROOT / "allennlp"
    TOOLS_ROOT = MODULE_ROOT / "tools"
    TESTS_ROOT = MODULE_ROOT / "tests"
    FIXTURES_ROOT = TESTS_ROOT / "fixtures"

    def setUp(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.DEBUG
        )
        # Disabling some of the more verbose logging statements that typically aren't very helpful
        # in tests.
        logging.getLogger("allennlp.common.params").disabled = True
        logging.getLogger("allennlp.nn.initializers").disabled = True
        logging.getLogger("allennlp.modules.token_embedders.embedding").setLevel(logging.INFO)
        logging.getLogger("urllib3.connectionpool").disabled = True
        log_pytorch_version_info()

        self.TEST_DIR = pathlib.Path(TEST_DIR)

        os.makedirs(self.TEST_DIR, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.TEST_DIR)


def parametrize(arg_names: Iterable[str], arg_values: Iterable[Iterable[Any]]):
    """
    Decorator to create parameterized tests.

    # Parameters

    arg_names : `Iterable[str]`, required.
        Argument names to pass to the test function.
    arg_values : `Iterable[Iterable[Any]]`, required.
        Iterable of values to pass to each of the args.
        The decorated test will be run for each inner iterable.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            for arg_value in arg_values:
                kwargs_extra = {name: value for name, value in zip(arg_names, arg_value)}
                func(*args, **kwargs, **kwargs_extra)

        return wrapper

    return decorator


_available_devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
multi_device = parametrize(("device",), [(device,) for device in _available_devices])
"""
Decorator that provides an argument `device` of type `str` for each available PyTorch device.
"""
