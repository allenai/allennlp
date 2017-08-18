import logging
import re
from typing import Callable, Dict, Sequence, Type

import torch
import torch.nn.init

from allennlp.common import Registrable
from allennlp.common.params import Params

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Initializer(Registrable):
    """
    An initializer is really just a bare pytorch function. This class
    is a proxy that allows us to implement ``Registerable`` for those functions.
    """
    default_implementation = 'normal'

    def __call__(self, tensor: torch.autograd.Variable) -> None:
        """
        This function is here just to make mypy happy.  We expect initialization functions to
        follow this API; the builtin pytorch initialization functions follow this just fine, even
        though they don't subclass ``Initialization``.  We're just making it explicit here, so mypy
        knows that initializers are callable like this.
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params):
        # Just a string - corresponds to the name of an initializer.
        if isinstance(params, str):
            return cls.by_name(params)()
        else:
            choice = params.pop_choice("type", cls.list_available())
            return cls.by_name(choice).from_params(params)


def _initializer_wrapper(init_function: Callable[..., None]) -> Type[Initializer]:
    class Init(Initializer):
        def __init__(self, **kwargs):
            self._init_function = init_function
            self._kwargs = kwargs
        def __call__(self, tensor: torch.autograd.Variable) -> None:
            self._init_function(tensor, **self._kwargs)
        def __repr__(self):
            return 'Init: %s, with params: %s' % (self._init_function, self._kwargs)
        @classmethod
        def from_params(cls, params: Params):
            return cls(**params.as_dict())
    return Init


# There are no classes to decorate, so we hack these into Registrable._registry
Registrable._registry[Initializer] = {  # pylint: disable=protected-access
        "normal": _initializer_wrapper(torch.nn.init.normal),
        "uniform": _initializer_wrapper(torch.nn.init.uniform),
        "orthogonal": _initializer_wrapper(torch.nn.init.orthogonal),
        "constant": _initializer_wrapper(torch.nn.init.constant),
        "dirac": _initializer_wrapper(torch.nn.init.dirac),
        "xavier_normal": _initializer_wrapper(torch.nn.init.xavier_normal),
        "xavier_uniform": _initializer_wrapper(torch.nn.init.xavier_uniform),
        "kaiming_normal": _initializer_wrapper(torch.nn.init.kaiming_normal),
        "kaiming_uniform": _initializer_wrapper(torch.nn.init.kaiming_uniform),
        "sparse": _initializer_wrapper(torch.nn.init.sparse),
        "eye": _initializer_wrapper(torch.nn.init.eye),
}


class InitializerApplicator:
    """
    Applies initializers to the parameters of a Module based on regex matches.
    All parameters in the Module will be initialized.
    """
    def __init__(self,
                 initializers: Dict[str, Initializer] = None,
                 default_initializer: Initializer = Initializer.by_name('normal')(),
                 exclude: Sequence[str] = None) -> None:
        """
        Parameters
        ----------
        initializers : ``Dict[str, Callable[[torch.Tensor], None]]``, optional (default = {})
            A dictionary mapping parameter regexes to initializers to be applied to parameters
            matching the regex.
        default_initializer : ``Callable[[torch.Tensor], None]``, optional (default = torch.nn.init.normal)
            A default initializer, which will be used in the case that the Applicator encounters a parameter
            which does not match any of the regexes provided.
        exclude : ``Sequence[str]``, optional (default=``[]``)
            A set of regexes for parameters that should be excluded from the default
            initialization.  This does *not* apply to the regexes passed in the ``initializers``
            parameter; it only filters the list of parameters that would otherwise get initialized
            by the default initializer.
        """
        self._initializers = initializers or {}
        self._default_initializer = default_initializer
        self._exclude = exclude or []

    def __call__(self, module: torch.nn.Module) -> None:
        """
        Applies a series of initializers to all parameters in a module if those parameters match a
        regex. If no explicitly specified initializers are applied, a default initializer is applied.

        Parameters
        ----------
        module : torch.nn.Module, required.
            The Pytorch module to apply the initializers to.
        """
        logger.info("Initializing parameters; finding explicit regex matches first")
        # Store which initialisers were applied to which parameters.
        not_explicitly_initialized_parameters = []
        for name, parameter in module.named_parameters():
            is_initialized = False
            for initializer_regex, initializer in self._initializers.items():
                if re.search(initializer_regex, name):
                    initializer(parameter)
                    logger.info("Initializing %s using %s intitializer", name, initializer_regex)
                    is_initialized = True
                    break
            if not is_initialized:
                not_explicitly_initialized_parameters.append((name, parameter))

        logger.info("Initializing remaining parameters with default initializer: %s",
                    self._default_initializer)
        for name, parameter in not_explicitly_initialized_parameters:
            if any(re.search(exclude_regex, name) for exclude_regex in self._exclude):
                logger.info("Excluding %s from default initialization", name)
            else:
                logger.info("Initializing %s using the default initializer", name)
                self._default_initializer(parameter)

    @classmethod
    def from_params(cls, params: Params) -> "InitializerApplicator":
        """
        Converts a Params object into an InitializerApplicator. The json should
        be formatted as follows::

            {
                "parameter_regex_match1": {
                    "type": "normal"
                    "mean": 0.01
                    "std": 0.1
                },
                "parameter_regex_match2": "uniform",
                "default": "orthogonal",
                "exclude": ["exclude_regex"]
            }

        where the keys are regex matches to the parameters, with the exception of the "default" and
        "exclude" keys.  The "default" key defines an initializer which will be used as the default
        initializer for parameters which do not match any initializer regex passed to the
        InitializerApplicator, except for any parameter with a name matching a regex in "exclude".

        The values for parameter regexes and for the "default" key will be passed to
        ``Initializer.from_params()``.  These values can either be strings, in which case they
        correspond to the names of initializers, or dictionaries, in which case they must contain
        the "type" key, corresponding to the name of an initializer.  In addition, they may contain
        auxiliary named parameters which will be fed to the initializer itself. To determine valid
        auxiliary parameters, please refer to the torch.nn.init documentation.

        Parameters
        ----------
        params: Params, required.
            A Params object containing an "initializers" key.

        Returns
        -------
        An InitializerApplicator containing the specified initializers.
        """
        exclude_regexes = params.pop("exclude", [])
        initializers = {}
        for name in list(params.keys()):
            initializer_params = params.pop(name)
            if name[0] == '"' and name[-1] == '"':
                name = name[1:-1]
            initializers[name] = Initializer.from_params(initializer_params)
        default = initializers.pop("default", Initializer.by_name('normal')())
        params.assert_empty(cls.__name__)
        return InitializerApplicator(initializers, default, exclude_regexes)
