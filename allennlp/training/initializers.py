from typing import List
import re

import torch.nn.init

from ..common.checks import ConfigurationError


class Initializer:
    """
    An abstract class representing an initializer. When called on a module,
    all parameters within that module should be initialised to some value
    in place. Modules matching a regex will be skipped.
    """
    def __init__(self, module_regex: str):
        self.module_regex = module_regex

    def __call__(self, module: torch.nn.Module):
        raise NotImplementedError

    def module_name_matches_regex(self, module: torch.nn.Module):
        return re.match(self.module_regex, module.__class__.__name__) is not None


class InitializerApplicator:
    """
    Applies a list of initializers to a model or Module recursively.
    All parameters in the Module will be initialized.
    """
    def __init__(self, initializers: List[Initializer]):
        self._initializers = initializers

        if all([x.module_regex == '' for x in initializers]) and len(self._initializers) > 1:
            raise ConfigurationError("No module_regex specified with multiple initializers causes"
                                     "all parameters to be set using the last initializer.")

    def _apply(self, module: torch.nn.Module, initializer: Initializer):
        for child in module.children():
            initializer(child)
            self._apply(child, initializer)

    def __call__(self, module: torch.nn.Module):
        for initializer in self._initializers:
            self._apply(module, initializer)


class Normal(Initializer):
    def __init__(self, mean: float = 0.0, std: float = 0.02, module_regex: str = ''):
        self.mean = mean
        self.std = std
        super(Normal, self).__init__(module_regex)

    def __call__(self, module: torch.nn.Module):
        if self.module_name_matches_regex(module):
            for parameter in module.parameters():
                torch.nn.init.normal(parameter, mean=self.mean, std=self.std)


class Uniform(Initializer):
    def __init__(self, a: float = 0.0, b: float = 1.0, module_regex: str = ''):
        self.a = a
        self.b = b
        super(Uniform, self).__init__(module_regex)

    def __call__(self, module: torch.nn.Module):
        if self.module_name_matches_regex(module):
            for parameter in module.parameters():
                torch.nn.init.uniform(parameter, a=self.a, b=self.b)


class Constant(Initializer):
    def __init__(self, value: float, module_regex: str = ''):
        self.value = value

        super(Constant, self).__init__(module_regex)

    def __call__(self, module: torch.nn.Module):
        if self.module_name_matches_regex(module):
            for parameter in module.parameters():
                torch.nn.init.constant(parameter, val=self.value)


class XavierUniform(Initializer):
    def __init__(self, gain: float = 1.0, module_regex: str = ''):
        self.gain = gain
        super(XavierUniform, self).__init__(module_regex)

    def __call__(self, module):
        if self.module_name_matches_regex(module):
            for parameter in module.parameters():
                torch.nn.init.xavier_uniform(parameter, gain=self.gain)


class XavierNormal(Initializer):
    def __init__(self, gain: float = 1.0, module_regex: str = ''):
        self.gain = gain

        super(XavierNormal, self).__init__(module_regex)

    def __call__(self, module: torch.nn.Module):
        if self.module_name_matches_regex(module):
            for parameter in module.parameters():
                torch.nn.init.xavier_normal(parameter, gain=self.gain)


class Orthogonal(Initializer):
    def __init__(self, gain: float = 1.0, module_regex: str = ''):
        self.gain = gain
        super(Orthogonal, self).__init__(module_regex)

    def __call__(self, module: torch.nn.Module):
        if self.module_name_matches_regex(module):
            for parameter in module.parameters():
                torch.nn.init.orthogonal(parameter, gain=self.gain)


class NormalSparse(Initializer):
    def __init__(self, sparsity: float, std: float = 0.01, module_regex: str = ''):
        self.sparsity = sparsity
        self.std = std
        super(NormalSparse, self).__init__(module_regex)

    def __call__(self, module: torch.nn.Module):
        if self.module_name_matches_regex(module):
            for parameter in module.paramters():
                torch.nn.init.sparse(parameter, sparsity=self.sparsity, std=self.std)
