from typing import List, Union

import torch

from allennlp.common.params import Params  # pylint: disable=unused-import
from allennlp.experiments import Registry


def get_optimizer_from_params(model_parameters: List[torch.Tensor],
                              params: Union[Params, str]) -> torch.optim.Optimizer:
    """
    This function converts a set of model parameters to be optimized and a
    dictionary of optimizer parameters (or a string name of one of the optimizers above)
    into a Pytorch optimizer object which can apply gradient updates to the model
    parameters following ``model.backward()`` calls.
    The simplest case for both of these is a string that shows up in `optimizers`
    above - if ``params`` is just one of those strings, we return it, and everyone
    is happy. If not, we assume ``params`` is a Dict[str, Any], with a "type" key,
    where the value for "type" must be one of those strings above. We take the rest
    of the parameters and pass them to the optimizer's constructor.
    """
    if isinstance(params, str):
        optimizer = params
        params = Params({})
    else:
        optimizer = params.pop_choice("type", Registry.list_optimizers(),
                                      default_to_first_choice=True)
    return Registry.get_optimizer(optimizer)(model_parameters, **params.as_dict())
