"""
AllenNLP just uses
`PyTorch optimizers <http://pytorch.org/docs/master/optim.html>`_ ,
with a thin wrapper to allow registering them and instantiating them ``from_params``.

The available optimizers are

* `"adadelta" <http://pytorch.org/docs/master/optim.html#torch.optim.Adadelta>`_
* `"adagrad" <http://pytorch.org/docs/master/optim.html#torch.optim.Adagrad>`_
* `"adam" <http://pytorch.org/docs/master/optim.html#torch.optim.Adam>`_
* `"sgd" <http://pytorch.org/docs/master/optim.html#torch.optim.SGD>`_
* `"rmsprop <http://pytorch.org/docs/master/optim.html#torch.optim.RMSprop>`_
"""

from typing import List

import torch

from allennlp.common import Params, Registrable


class Optimizer(Registrable):
    """
    This class just allows us to implement ``Registrable`` for Pytorch Optimizers.
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
