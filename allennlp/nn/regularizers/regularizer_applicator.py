import re
from typing import Sequence, Tuple, List, Optional

import torch

from allennlp.common.params import Params
from allennlp.nn.regularizers.regularizer import Regularizer


class RegularizerApplicator:
    """
    Applies regularizers to the parameters of a Module based on regex matches.
    """
    def __init__(self, regularizers: Sequence[Tuple[str, Regularizer]] = ()) -> None:
        """
        Parameters
        ----------
        regularizers : Sequence[Tuple[str, Regularizer]], optional (default = ())
            A sequence of pairs (regex, Regularizer), where each Regularizer
            applies to the parameters its regex matches (and that haven't previously
            been matched).
        """
        self._regularizers = regularizers

    def __call__(self, module: torch.nn.Module) -> torch.Tensor:
        """
        Parameters
        ----------
        module : torch.nn.Module, required
            The module to regularize.
        """
        accumulator = 0.0
        # For each parameter find the first matching regex.
        for name, parameter in module.named_parameters():
            for regex, regularizer in self._regularizers:
                if re.search(regex, name):
                    penalty = regularizer(parameter)
                    accumulator = accumulator + penalty
                    break

        return accumulator

    @classmethod
    def from_params(cls, params: List[Tuple[str, Params]]) -> Optional['RegularizerApplicator']:
        """
        Converts a List of pairs (regex, params) into an RegularizerApplicator.
        This list should look like

        [["regex1": {"type": "l2", "alpha": 0.01}], ["regex2": "l1"]]

        where each parameter receives the penalty corresponding to the first regex
        that matches its name (which may be no regex and hence no penalty).
        The values can either be strings, in which case they correspond to the names
        of regularizers, or dictionaries, in which case they must contain the "type"
        key, corresponding to the name of a regularizer. In addition, they may contain
        auxiliary named parameters which will be fed to the regularizer itself.
        To determine valid auxiliary parameters, please refer to the torch.nn.init documentation.

        Parameters
        ----------
        params : ``Params``, required.
            A Params object containing a "regularizers" key.

        Returns
        -------
        A RegularizerApplicator containing the specified Regularizers,
        or ``None`` if no Regularizers are specified.
        """
        if not params:
            return None

        instantiated_regularizers = []
        for parameter_regex, regularizer_params in params:
            if isinstance(regularizer_params, str):
                regularizer = Regularizer.by_name(regularizer_params)()
            else:
                regularizer_type = Regularizer.by_name(regularizer_params.pop("type"))
                regularizer = regularizer_type(**regularizer_params)  # type: ignore
            instantiated_regularizers.append((parameter_regex, regularizer))
        return RegularizerApplicator(instantiated_regularizers)
