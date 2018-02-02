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

import re
from typing import List

import torch

from allennlp.common import Params, Registrable


class Optimizer(Registrable):
    """
    This class just allows us to implement ``Registrable`` for Pytorch Optimizers.
    """
    default_implementation = "adam"

    @classmethod
    def from_params(cls, model_parameters: List, params: Params):
        if isinstance(params, str):
            optimizer = params
            params = Params({})
        else:
            optimizer = params.pop_choice("type", Optimizer.list_available())

        # make the parameter groups if need
        groups = params.pop("parameter_groups", None)
        if groups:
            # input to optimizer is list of dict
            # each dict contains {'params': [list of parameters], 'lr': 1e-3, ...}
            # see: http://pytorch.org/docs/0.3.0/optim.html?#per-parameter-options
            #
            # groups contains something like:
            #"parameter_groups": [
            #       [['regex1', 'regex2'], {'lr': 1e-3},
            #        ['regex3'], {'lr': 1e-4}]
            #]
            # last entry is for the parameters not in any regex
            parameter_groups = [{'params': []} for _ in range(len(groups) + 1)]
            # add the group specific kwargs
            for k in range(len(groups)):
                parameter_groups[k].update(groups[k][1].as_dict())

            for name, param in model_parameters:
                # Determine the group for this parameter.
                group_index = None
                for k, group_regexes in enumerate(groups):
                    for regex in group_regexes[0]:
                        if re.search(regex, name):
                            if group_index is not None and group_index != k:
                                raise ValueError("{} was specified in two separate parameter groups".format(name))
                            group_index = k

                if group_index is not None:
                    parameter_groups[k]['params'].append(param)
                else:
                    # the default group
                    parameter_groups[-1]['params'].append(param)
        else:
            parameter_groups = [param for name, param in model_parameters]

        return Optimizer.by_name(optimizer)(parameter_groups, **params.as_dict()) # type: ignore

# We just use the Pytorch optimizers, so here we force them into
# Registry._registry so we can build them from params.
Registrable._registry[Optimizer] = {   # pylint: disable=protected-access
        "adam": torch.optim.Adam,
        "adagrad": torch.optim.Adagrad,
        "adadelta": torch.optim.Adadelta,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
}
