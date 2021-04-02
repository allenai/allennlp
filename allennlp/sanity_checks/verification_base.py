"""
Code based almost entirely on
https://github.com/awaelchli/pytorch-lightning-snippets/commit/7db53f774715d635c59ef56f21a17634d246b2c5
"""

from abc import abstractmethod
from copy import deepcopy

import torch.nn as nn
from typing import Any
from allennlp.nn.util import move_to_device


class VerificationBase:
    """
    Base class for model verification.
    All verifications should run with any :class:`torch.nn.Module` unless otherwise stated.
    """

    def __init__(self, model: nn.Module):
        """
        Arguments:
            model: The model to run verification for.
        """
        super().__init__()
        self.model = model

    @abstractmethod
    def check(self, *args, **kwargs) -> bool:
        """Runs the actual test on the model. All verification classes must implement this.
        Arguments:
            *args: Any positional arguments that are needed to run the test
            *kwargs: Keyword arguments that are needed to run the test
        Returns:
            `True` if the test passes, and `False` otherwise. Some verifications can only be performed
            with a heuristic accuracy, thus the return value may not always reflect the true state of
            the system in these cases.
        """
        pass

    def _get_inputs_copy(self, inputs) -> Any:
        """
        Returns a deep copy of the example inputs in cases where it is expected that the
        input changes during the verification process.
        Arguments:
            inputs: The inputs to clone.
        """
        inputs = deepcopy(inputs)
        inputs = move_to_device(inputs, device=next(self.model.parameters()).device)

        return inputs

    def _model_forward(self, inputs: Any) -> Any:
        """
        Feeds the inputs to the model via the ``__call__`` method.
        Arguments:
            inputs: The input that goes into the model. If it is a tuple, it gets
                interpreted as the sequence of positional arguments and is passed in by tuple unpacking.
                If it is a dict, the contents get passed in as named parameters by unpacking the dict.
                Otherwise, the input array gets passed in as a single argument.
        Returns:
            The output of the model.
        """
        if isinstance(inputs, tuple):
            return self.model(*inputs)
        if isinstance(inputs, dict):
            return self.model(**inputs)
        return self.model(inputs)
