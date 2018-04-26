"""
Functions and exceptions for checking that
AllenNLP and its models are configured correctly.
"""
from typing import Callable
import logging

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

_MUST_OVERRIDE = "__MUST_OVERRIDE"

def must_override_method(method: Callable):
    """
    Decorator that indicates that a method must be overridden in a subclass.
    It does this by setting an attribute on the method itself, which then works
    in combination with the ``check_is_overridden`` method below.
    """
    setattr(method, _MUST_OVERRIDE, True)
    return method

def check_is_overridden(method: Callable, name: str = ''):
    """
    Makes sure that a method does not have its "must override" attribute set,
    which it only will if you used the ``@must_override_method`` decorator
    on the base class method and are looking at the base class method.
    """
    if hasattr(method, _MUST_OVERRIDE):
        raise ConfigurationError(f"You must override {name} in your subclass")
