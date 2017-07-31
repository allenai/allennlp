from typing import List

import torch

from allennlp.common.params import Params  # pylint: disable=unused-import
from allennlp.common.registrable import Registrable


class Optimizer(Registrable):
    """
    This class just allows us to implement ``Registerable`` for Pytorch Optimizers.
    """
    default_implementation = "adam"

    @classmethod
    def from_params(cls, model_parameters: List[torch.nn.Parameter], params: Params):
        if isinstance(params, str):
            optimizer = params
            params = Params({})
        else:
            optimizer = params.pop_choice("type", Optimizer.list_available())
        return Optimizer.by_name(optimizer)(model_parameters, **params.as_dict()) # type: ignore

# We just use the Pytorch optimizers, so here we force them into
# Registry._registry so we can build them from params.
Registrable._registry[Optimizer] = {   # pylint: disable=protected-access
        "adam": torch.optim.Adam,
        "adagrad": torch.optim.Adagrad,
        "adadelta": torch.optim.Adadelta,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
}
