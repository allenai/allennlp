from typing import Any, Dict, List, Union

import torch

from ..common.checks import ConfigurationError

optimizers = {
    "adam": torch.optim.Adam,
    "adagrad": torch.optim.Adagrad,
    "adadelta": torch.optim.Adadelta,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,

}


def get_optimizer_from_params(model_parameters: List[torch.Tensor],
                              params: Union[Dict[str, Any], str]):
    """
    This function converts a set of model parameters to be optimized and a
    dictionary of optimizer parameters (or a string name of one of the optimizers above)
    into a Pytorch optimizer object which can apply gradient updates to the model
    parameters following model.backward() calls.
    The simplest case for both of these is a string that shows up in `optimizers`
    above - if `params` is just one of those strings, we return it, and everyone
    is happy. If not, we assume `params` is a Dict[str, Any], with a "type" key,
    where the value for "type" must be one of those strings above. We take the rest
    of the parameters and pass them to the optimizer's constructor.

    """
    if isinstance(params, str):
        optimizer = params
        params = {}
    else:
        optimizer = params.pop("type")
    if optimizer is None or optimizer not in optimizers.keys():
        raise ConfigurationError("{} not in allowed optimizers {}".format(optimizer, " ".join(optimizers.keys())))
    return optimizers[optimizer](model_parameters, **params)
