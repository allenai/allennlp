import re
from typing import List, Tuple

import torch

from allennlp.common import FromParams
from allennlp.nn.regularizers.regularizer import Regularizer


class RegularizerApplicator(FromParams):
    """
    Applies regularizers to the parameters of a Module based on regex matches.
    """

    def __init__(self, regexes: List[Tuple[str, Regularizer]] = None) -> None:
        """
        # Parameters

        regexes : `List[Tuple[str, Regularizer]]`, optional (default = `None`)
            A sequence of pairs (regex, Regularizer), where each Regularizer
            applies to the parameters its regex matches (and that haven't previously
            been matched).
        """
        self._regularizers = regexes or []

    def __call__(self, module: torch.nn.Module) -> torch.Tensor:
        """
        # Parameters

        module : `torch.nn.Module`, required
            The module to regularize.
        """
        accumulator = 0.0
        for name, parameter in module.named_parameters():
            # We first check if the parameter needs gradient updates or not
            if parameter.requires_grad:
                # For each parameter find the first matching regex.
                for regex, regularizer in self._regularizers:
                    if re.search(regex, name):
                        penalty = regularizer(parameter)
                        accumulator = accumulator + penalty
                        break
        return accumulator
