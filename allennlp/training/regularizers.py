from typing import List
import re

import torch


class Regularizer:
    """
    Abstract class representing some regularisation function
    applied to a Module. When called, it returns a scalar float,
    which is typically applied to a loss function.
    """
    def __init__(self, module_regex: str = ""):
        self.module_regex = module_regex

    def __call__(self, module: torch.nn.Module) -> float:
        raise NotImplementedError

    def module_name_matches_regex(self, module: torch.nn.Module):
        return re.match(self.module_regex, module.__class__.__name__) is not None


class RegularizerApplicator:
    """
    Recursively applies a list of regularizers to a Module
    and it's children.
    """
    def __init__(self, regluarizers: List[Regularizer]):

        self._regularizers = regluarizers

        self.accumulator = 0.

    def _regularize_module(self,
                           module: torch.nn.Module,
                           regularizer: Regularizer):

        for child in module.children():
                self.accumulator += regularizer(child)
                self._regularize_module(child, regularizer)

    def __call__(self, module: torch.nn.Module) -> float:
        self.accumulator = 0.
        for regularizer in self._regularizers:
            self._regularize_module(module, regularizer)

        regularization_value = self.accumulator
        self.accumulator = 0.
        return regularization_value


class L1Regularizer(Regularizer):

    def __init__(self, alpha: float = 0.01, module_regex: str = ""):
        self.alpha = alpha
        super(L1Regularizer, self).__init__(module_regex=module_regex)

    def __call__(self, module: torch.nn.Module) -> float:
        value = 0.0
        if self.module_name_matches_regex(module):
            for parameter in module.parameters():
                value += torch.sum(torch.abs(parameter))

        return self.alpha * value


class L2Regularizer(Regularizer):

    def __init__(self, alpha: float = 0.01, module_regex: str = ""):
        self.alpha = alpha
        super(L2Regularizer, self).__init__(module_regex=module_regex)

    def __call__(self, module: torch.nn.Module) -> float:
        value = 0.0
        if self.module_name_matches_regex(module):
            for parameter in module.parameters():
                value += torch.sum(torch.pow(parameter, 2))
        return self.alpha * value
