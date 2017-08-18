import re
from typing import Dict

import torch

from allennlp.common.params import Params
from allennlp.training.regularizers.regularizer import Regularizer


class RegularizerApplicator:
    """
    Applies regularizers to the parameters of a Module based on regex matches.
    """
    def __init__(self, regularizers: Dict[str, Regularizer]) -> None:
        """
        Parameters
        ----------
        regularizers : Dict[str, Callable[[torch.Tensor], None]], optional (default = None)
            A dictionary mapping parameter regexes to regularizers to be applied to parameters
            matching the regex.
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
        for parameter_regex, regularizer in self._regularizers.items():
            for name, parameter in module.named_parameters():
                if re.search(parameter_regex, name):
                    accumulator += regularizer(parameter)
        return accumulator

    @classmethod
    def from_params(cls, params: Params) -> 'RegularizerApplicator':
        """
        Converts a Params object into an RegularizerApplicator. The json should
        be formatted as follows::

            regularizers: {
                parameter_regex_match1: {
                    "type": "l2"
                    "alpha": 0.01
                },
                parameter_regex_match2: "l1",
            }

        where the keys are regex matches to parameter names. The values can either be strings,
        in which case they correspond to the names of regularizers, or dictionaries, in which
        case they must contain the "type" key, corresponding to the name of a regularizer.
        In addition, they may contain auxiliary named parameters which will be fed to the
        regularizer itself. To determine valid auxiliary parameters, please refer to the
        torch.nn.init documentation.

        Parameters
        ----------
        params: Params, required.
            A Params object containing a "regularizers" key.

        Returns
        -------
        A RegularizerApplicator containing the specified Regularizers.
        """
        all_regularizer_params = params.pop("regularizers", {}).as_dict()

        instantiated_regularizers = {}
        for parameter_regex, regularizer_params in all_regularizer_params.items():
            if isinstance(regularizer_params, str):
                instantiated_regularizers[parameter_regex] = Regularizer.by_name(regularizer_params)()
            else:
                regularizer_type = Regularizer.by_name(regularizer_params.pop("type"))
                instantiated_regularizers[parameter_regex] = regularizer_type(**regularizer_params)  # type: ignore
        return RegularizerApplicator(instantiated_regularizers)
