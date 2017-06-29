from typing import List
import re

import torch.nn.init

from ..common.checks import ConfigurationError


class Initializer:

    def __init__(self, module_regex: str):
        self.module_regex = module_regex

    def __call__(self, module: torch.nn.Module):
        raise NotImplementedError

    def module_name_matches_regex(self, module: torch.nn.Module):
        return re.match(self.module_regex, module.__class__.__name__) is not None


class InitializerApplicator:
    def __init__(self, initializers: List[Initializer]):
        self._initializers = initializers

        if all([x.module_regex == '' for x in initializers]) and len(self._initializers) > 1:
            raise ConfigurationError("No module_regex specified with multiple initializers causes"
                                     "all parameters to be set using the last initializer.")

    def _apply(self, module: torch.nn.Module, initializer: Initializer):
        for child in module.children():
            initializer(child)
            self._apply(child, initializer)

    def __call__(self, module):
        for initializer in self._initializers:
            self._apply(module, initializer)


class Normal(Initializer):
    def __init__(self, mean=0.0, std=0.02, module_regex=''):
        self.mean = mean
        self.std = std
        super(Normal, self).__init__(module_regex)

    def __call__(self, module):
        if self.module_name_matches_regex(module):
            for parameter in module.parameters():
                torch.nn.init.normal(parameter, mean=self.mean, std=self.std)


class Uniform(Initializer):
    def __init__(self, a=0, b=1, module_regex=''):
        self.a = a
        self.b = b
        super(Uniform, self).__init__(module_regex)

    def __call__(self, module):
        if self.module_name_matches_regex(module):
            for parameter in module.parameters():
                torch.nn.init.uniform(parameter, a=self.a, b=self.b)


class Constant(Initializer):
    def __init__(self, value, module_regex=''):
        self.value = value

        super(Constant, self).__init__(module_regex)

    def __call__(self, module):
        if self.module_name_matches_regex(module):
            for parameter in module.parameters():
                torch.nn.init.constant(parameter, val=self.value)


class XavierUniform(Initializer):
    def __init__(self, gain=1, module_regex=''):
        self.gain = gain
        super(XavierUniform, self).__init__(module_regex)

    def __call__(self, module):
        if self.module_name_matches_regex(module):
            for parameter in module.parameters():
                torch.nn.init.xavier_uniform(parameter, gain=self.gain)


class XavierNormal(Initializer):
    def __init__(self, gain=1, module_regex=''):
        self.gain = gain

        super(XavierNormal, self).__init__(module_regex)

    def __call__(self, module):
        if self.module_name_matches_regex(module):
            for parameter in module.parameters():
                torch.nn.init.xavier_normal(parameter, gain=self.gain)


class Orthogonal(Initializer):
    def __init__(self, gain=1, module_regex=''):
        self.gain = gain
        super(Orthogonal, self).__init__(module_regex)

    def __call__(self, module):
        if self.module_name_matches_regex(module):
            for parameter in module.parameters():
                torch.nn.init.orthogonal(parameter, gain=self.gain)


class NormalSparse(Initializer):
    def __init__(self, sparsity, std=0.01, module_regex=''):
        self.sparsity = sparsity
        self.std = std
        super(NormalSparse, self).__init__(module_regex)

    def __call__(self, module):
        if self.module_name_matches_regex(module):
            for parameter in module.paramters():
                torch.nn.init.sparse(parameter, sparsity=self.sparsity, std=self.std)
