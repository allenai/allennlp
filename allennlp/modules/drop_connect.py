from typing import Iterator
import re
import warnings

import torch
import torch.nn.functional as F


class _Match:
    """
    Helper object that stores ``Parameter`` objects along with their parent ``Module`` objects
    and associated names.
    """

    def __init__(
        self,
        module_name: str,
        module: torch.nn.Module,
        parameter_name: str,
        parameter: torch.nn.Parameter,
    ) -> None:
        self.module_name = module_name
        self.module = module
        self.parameter_name = parameter_name
        self.parameter = parameter

    @property
    def raw_parameter_name(self):
        prefix = self.module_name.replace(
            ".", "_"
        )  # PyTorch will complain if the name contains a '.'
        return "_".join((prefix, self.parameter_name)) + "_raw"


class DropConnect(torch.nn.Module):
    """
    DropConnect module described in: `"Regularization of Neural Networks using DropConnect"
    <https://www.semanticscholar.org/paper/Regularization-of-Neural-Networks-using-DropConnect-Wan-Zeiler/38f35dd624cd1cf827416e31ac5e0e0454028eca>`_
    by Wan et al., 2013. Applies dropout to module parameters instead of module outputs.

    Parameters
    ==========
    module : ``torch.nn.Module``
        Module to apply weight dropout to.
    parameter_regex : ``str``
        Regular expression identifying which parameters to apply weight dropout to.
    dropout : ``float``, optional (default = 0.0)
        Probability that a given weight is dropped.
    """

    def __init__(self, module: torch.nn.Module, parameter_regex: str, dropout: float = 0.0) -> None:
        super().__init__()
        self._module = module
        self._parameter_regex = parameter_regex
        self._dropout = dropout
        # Find all of the parameters that match the regular expression and cache them, the
        # submodule they belong to, as well as their names. We need to know the specific submodule
        # so that we can place the dropped out tensors into the correct Module._parameters
        # dictionaries.
        self._matches = list(self._search_parameters(self._parameter_regex))
        # Register the raw parameters (e.g., the parameters before dropout is applied) to this
        # module and remove them from their parent module. This avoids double counting when using
        # the Module.parameters() method. If we didn't do this the matched parameters would be
        # updated twice when calling Optimizer.step().
        for match in self._matches:
            self.register_parameter(match.raw_parameter_name, match.parameter)
            delattr(match.module, match.parameter_name)
        # Skip `RNNBase._apply()`'s last step otherwise it will run into mismatched `_all_weights`.`
        # Use `_called_no_op_f_called_no_op_flatten_parameters` as a dummy flag for test coverage.
        self._called_no_op_flatten_parameters = None
        if issubclass(type(self._module), torch.nn.RNNBase):
            self._module.flatten_parameters = self._no_op_flatten_parameters
            self._called_no_op_flatten_parameters = 0

    def _search_parameters(self, regex: str) -> Iterator[_Match]:
        """
        Generates _Match objects for all parameters whose names match the provided regular
        expression.
        """
        for submodule_name, submodule in self.named_modules():
            for parameter_name, parameter in submodule.named_parameters(recurse=False):
                if re.search(regex, parameter_name):
                    yield _Match(submodule_name, submodule, parameter_name, parameter)

    def _no_op_flatten_parameters(self):
        """
        We need to replace flatten_parameters with a no-op function.
        It must be a function rather than a lambda as otherwise pickling explodes.
        """
        if self._called_no_op_flatten_parameters is not None:
            self._called_no_op_flatten_parameters += 1
        return

    def forward(self, *args):
        # Apply dropout to all of the matching parameters, and force them into their original
        # parent module's _parameter dictionary
        for match in self._matches:
            raw_parameter = getattr(self, match.raw_parameter_name)
            match.module._parameters[match.parameter_name] = F.dropout(
                raw_parameter, p=self._dropout, training=self.training
            )
        # Avoid the warning of
        # `/pytorch/aten/src/ATen/native/cudnn/RNN.cpp`:1238:
        # `UserWarning`: RNN module weights are not part of single contiguous chunk of memory.
        # This means they need to be compacted at every call, possibly greatly increasing memory
        # usage. To compact weights again call `flatten_parameters()`.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Call the top-level module on the inputs.
            output = self._module.forward(*args)
        # Not removing the dropped out parameters from their parent module's _parameter dictionary.
        # Because PyTorch's optimizer won't include non-leaf tensors as parameters.

        return output
