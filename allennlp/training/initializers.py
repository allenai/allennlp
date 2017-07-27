from typing import Dict, Callable
import logging
import re

import torch.nn.init

from allennlp.common.params import Params
from allennlp.training.initializer import Initializer
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class InitializerApplicator:
    """
    Applies initializers to the parameters of a Module based on regex matches.
    All parameters in the Module will be initialized.
    """
    def __init__(self,
                 initializers: Dict[str, Callable[[torch.Tensor], None]] = None,
                 default_initializer: Callable[[torch.Tensor], None] = torch.nn.init.normal) -> None:
        """
        Parameters
        ----------
        initializers : Dict[str, Callable[[torch.Tensor], None]], optional (default = {})
            A dictionary mapping parameter regexes to initializers to be applied to parameters
            matching the regex.
        default_initializer : Callable[[torch.Tensor], None], optional, (default = torch.nn.init.normal)
            A default initializer, which will be used in the case that the Applicator encounters a parameter
            which does not match any of the regexes provided.
        """
        self._initializers = initializers or {}
        self._default_initializer = default_initializer

    def __call__(self, module: torch.nn.Module) -> None:
        """
        Applies a series of initializers to all parameters in a module if those parameters match a
        regex. If no explicitly specified initializers are applied, a default initializer is applied.

        Parameters
        ----------
        module : torch.nn.Module, required.
            The Pytorch module to apply the initializers to.
        """
        # Store which initialisers were applied to which parameters.
        not_explicitly_initialized_parameters = []
        for name, parameter in module.named_parameters():
            is_initialized = False
            for initializer_regex, initializer in self._initializers.items():
                if re.search(initializer_regex, name):
                    initializer(parameter)
                    logger.info("Initializing %s using %s "
                                "Intitializer.", name, initializer_regex)
                    is_initialized = True
            if not is_initialized:
                not_explicitly_initialized_parameters.append((name, parameter))

        for name, parameter in not_explicitly_initialized_parameters:
            self._default_initializer(parameter)
            logger.info("Initializing %s using the Default "
                        "Intitializer. (Normal(0, 1) unless user specified.)", name)

    @classmethod
    def from_params(cls, params: Params) -> "InitializerApplicator":
        """
        Converts a Params object into an InitializerApplicator. The json should
        be formatted as follows:

        initializers: {
            parameter_regex_match1: {
                "type": "normal"
                "mean": 0.01
                "std": 0.1
            },
            parameter_regex_match2: "uniform",

            "default": "orthogonal"
        }
        where the keys are regex matches to the parameters (with the exception of the "default" key,
        which will be used as the default initializer for parameters which do not match any
        initializer regex passed to the InitializerApplicator). The values can either be strings,
        in which case they correspond to the names of initializers, or dictionaries, in which
        case they must contain the "type" key, corresponding to the name of an initializer.
        In addition, they may contain auxiliary named parameters which will be fed to the
        initializer itself. To determine valid auxiliary parameters, please refer to the
        torch.nn.init documentation.

        Parameters
        ----------
        params: Params, required.
            A Params object containing an "initializers" key.

        Returns
        -------
        An InitializerApplicator containing the specified initializers.
        """
        # Construct a dictionary of available initializers from the torch.nn.init package.
        all_initializer_params = params.pop("initializers", {}).as_dict()
        instantiated_initializers = {}
        for name, initializer_params in all_initializer_params.items():
            # Just a string - corresponds to the name of an initializer.
            if isinstance(initializer_params, str):
                instantiated_initializers[name] = Initializer.by_name(initializer_params)
            else:
                initializer_type = initializer_params.pop("type")
                # This is to avoid passing by reference inside the curried function.
                # Without creating a new dict, we would pass the value of initializer_params
                # when it is called, which could be different as it is a loop variable.
                init_params = {**initializer_params}

                # pylint: disable=cell-var-from-loop
                def curried_initializer(tensor: torch.Tensor):
                    return Initializer.by_name(initializer_type)(tensor, **init_params)  # type: ignore
                # pylint: enable=cell-var-from-loop
                instantiated_initializers[name] = curried_initializer  # type: ignore
        try:
            default = instantiated_initializers.pop("default")
        except KeyError:
            default = torch.nn.init.normal
        return InitializerApplicator(instantiated_initializers, default)  # type: ignore
