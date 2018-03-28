"""
AllenNLP just uses
`PyTorch optimizers <http://pytorch.org/docs/master/optim.html>`_ ,
with a thin wrapper to allow registering them and instantiating them ``from_params``.

The available optimizers are

* `"adadelta" <http://pytorch.org/docs/master/optim.html#torch.optim.Adadelta>`_
* `"adagrad" <http://pytorch.org/docs/master/optim.html#torch.optim.Adagrad>`_
* `"adam" <http://pytorch.org/docs/master/optim.html#torch.optim.Adam>`_
* `"sparse_adam" <http://pytorch.org/docs/master/optim.html#torch.optim.SparseAdam>`_
* `"sgd" <http://pytorch.org/docs/master/optim.html#torch.optim.SGD>`_
* `"rmsprop <http://pytorch.org/docs/master/optim.html#torch.optim.RMSprop>`_
* `"adamax <http://pytorch.org/docs/master/optim.html#torch.optim.Adamax>`_
* `"averaged_sgd <http://pytorch.org/docs/master/optim.html#torch.optim.ASGD>`_
"""

import logging
import re
from typing import List, Any, Dict

import torch

from allennlp.common import Params, Registrable

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


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
            # The input to the optimizer is list of dict.
            # Each dict contains a "parameter group" and groups specific options,
            # e.g., {'params': [list of parameters], 'lr': 1e-3, ...}
            # Any config option not specified in the additional options (e.g.
            # for the default group) is inherited from the top level config.
            # see: http://pytorch.org/docs/0.3.0/optim.html?#per-parameter-options
            #
            # groups contains something like:
            #"parameter_groups": [
            #       [["regex1", "regex2"], {"lr": 1e-3},
            #        ["regex3"], {"lr": 1e-4}]
            #]
            #(note that the allennlp config files require double quotes ", and will
            # fail (sometimes silently) with single quotes ').

            # This is typed as as Any since the dict values other then
            # the params key are passed to the Optimizer constructor and
            # can be any type it accepts.
            # In addition to any parameters that match group specific regex,
            # we also need a group for the remaining "default" group.
            # Those will be included in the last entry of parameter_groups.
            parameter_groups: Any = [{'params': []} for _ in range(len(groups) + 1)]
            # add the group specific kwargs
            for k in range(len(groups)): # pylint: disable=consider-using-enumerate
                parameter_groups[k].update(groups[k][1].as_dict())

            regex_use_counts: Dict[str, int] = {}
            parameter_group_names: List[set] = [set() for _ in range(len(groups) + 1)]
            for name, param in model_parameters:
                # Determine the group for this parameter.
                group_index = None
                for k, group_regexes in enumerate(groups):
                    for regex in group_regexes[0]:
                        if regex not in regex_use_counts:
                            regex_use_counts[regex] = 0
                        if re.search(regex, name):
                            if group_index is not None and group_index != k:
                                raise ValueError("{} was specified in two separate parameter groups".format(name))
                            group_index = k
                            regex_use_counts[regex] += 1

                if group_index is not None:
                    parameter_groups[group_index]['params'].append(param)
                    parameter_group_names[group_index].add(name)
                else:
                    # the default group
                    parameter_groups[-1]['params'].append(param)
                    parameter_group_names[-1].add(name)

            # log the parameter groups
            logger.info("Done constructing parameter groups.")
            for k in range(len(groups) + 1):
                group_options = {key: val for key, val in parameter_groups[k].items()
                                 if key != 'params'}
                logger.info("Group %s: %s, %s", k,
                            list(parameter_group_names[k]),
                            group_options)
            # check for unused regex
            for regex, count in regex_use_counts.items():
                if count == 0:
                    logger.warning("When constructing parameter groups, "
                                   " %s not match any parameter name", regex)

        else:
            parameter_groups = [param for name, param in model_parameters]

        return Optimizer.by_name(optimizer)(parameter_groups, **params.as_dict()) # type: ignore

# We just use the Pytorch optimizers, so here we force them into
# Registry._registry so we can build them from params.
Registrable._registry[Optimizer] = {   # pylint: disable=protected-access
        "adam": torch.optim.Adam,
        "sparse_adam": torch.optim.SparseAdam,
        "adagrad": torch.optim.Adagrad,
        "adadelta": torch.optim.Adadelta,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "adamax": torch.optim.Adamax,
        "averaged_sgd": torch.optim.ASGD,
}
