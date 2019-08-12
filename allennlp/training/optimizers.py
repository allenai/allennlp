"""
AllenNLP just uses
`PyTorch optimizers <https://pytorch.org/docs/master/optim.html>`_ ,
with a thin wrapper to allow registering them and instantiating them ``from_params``.

The available optimizers are

* `"adadelta" <https://pytorch.org/docs/master/optim.html#torch.optim.Adadelta>`_
* `"adagrad" <https://pytorch.org/docs/master/optim.html#torch.optim.Adagrad>`_
* `"adam" <https://pytorch.org/docs/master/optim.html#torch.optim.Adam>`_
* `"sparse_adam" <https://pytorch.org/docs/master/optim.html#torch.optim.SparseAdam>`_
* `"sgd" <https://pytorch.org/docs/master/optim.html#torch.optim.SGD>`_
* `"rmsprop <https://pytorch.org/docs/master/optim.html#torch.optim.RMSprop>`_
* `"adamax <https://pytorch.org/docs/master/optim.html#torch.optim.Adamax>`_
* `"averaged_sgd <https://pytorch.org/docs/master/optim.html#torch.optim.ASGD>`_
"""

import logging
import re
import math
from typing import List, Any, Dict

import torch
from pytorch_pretrained_bert.optimization import BertAdam

from allennlp.common import Params, Registrable

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Optimizer(Registrable):
    """
    This class just allows us to implement ``Registrable`` for Pytorch Optimizers.
    """
    default_implementation = "adam"

    # Requires custom from_params.
    @classmethod
    def from_params(cls, model_parameters: List, params: Params):  # type: ignore
        # pylint: disable=arguments-differ
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
            # see: https://pytorch.org/docs/0.3.0/optim.html?#per-parameter-options
            #
            # groups contains something like:
            #"parameter_groups": [
            #       [["regex1", "regex2"], {"lr": 1e-3}],
            #       [["regex3"], {"lr": 1e-4}]
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

        # Log the number of parameters to optimize
        num_parameters = 0
        for parameter_group in parameter_groups:
            if isinstance(parameter_group, dict):
                num_parameters += sum(parameter.numel() for parameter in parameter_group["params"])
            else:
                num_parameters += parameter_group.numel()
        logger.info("Number of trainable parameters: %s", num_parameters)

        # By default we cast things that e.g. look like floats to floats before handing them
        # to the Optimizer constructor, but if you want to disable that behavior you could add a
        #       "infer_type_and_cast": false
        # key to your "trainer.optimizer" config.
        infer_type_and_cast = params.pop_bool("infer_type_and_cast", True)
        params_as_dict = params.as_dict(infer_type_and_cast=infer_type_and_cast)
        subclass = Optimizer.by_name(optimizer)

        # If the optimizer subclass has a from_params, use it.
        if hasattr(subclass, 'from_params'):
            return subclass.from_params(parameter_groups, params=params)
        else:
            return subclass(parameter_groups, **params_as_dict) # type: ignore

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
        "bert_adam": BertAdam,
}

def _safe_sparse_mask(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    In PyTorch 1.0, Tensor._sparse_mask was changed to Tensor.sparse_mask.
    This wrapper allows AllenNLP to (temporarily) work with both 1.0 and 0.4.1.
    """
    # pylint: disable=protected-access
    try:
        return tensor.sparse_mask(mask)
    except AttributeError:
        # TODO(joelgrus): remove this and/or warn at some point
        return tensor._sparse_mask(mask)


@Optimizer.register('dense_sparse_adam')
class DenseSparseAdam(torch.optim.Optimizer):
    # pylint: disable=protected-access,cell-var-from-loop
    # pylint: disable=unneeded-not,misplaced-comparison-constant
    # pylint: disable=len-as-condition,invalid-name,anomalous-backslash-in-string
    """
    NOTE: This class has been copied verbatim from the separate Dense and
    Sparse versions of Adam in Pytorch.

    Implements Adam algorithm with dense & sparse gradients.
    It has been proposed in Adam: A Method for Stochastic Optimization.

    Parameters
    ----------
    params : ``iterable``
        iterable of parameters to optimize or dicts defining parameter groups
    lr : ``float``, optional (default: 1e-3)
        The learning rate.
    betas : ``Tuple[float, float]``, optional (default: (0.9, 0.999))
        coefficients used for computing running averages of gradient
        and its square.
    eps : ``float``, optional, (default: 1e-8)
        A term added to the denominator to improve numerical stability.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(DenseSparseAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Parameters
        ----------
        closure : ``callable``, optional.
            A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['step'] += 1

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                if grad.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)

                    # Decay the first and second moment running average coefficient
                    #      old <- b * old + (1 - b) * new
                    # <==> old += (1 - b) * (new - old)
                    old_exp_avg_values = _safe_sparse_mask(exp_avg, grad)._values()
                    exp_avg_update_values = grad_values.sub(old_exp_avg_values).mul_(1 - beta1)
                    exp_avg.add_(make_sparse(exp_avg_update_values))
                    old_exp_avg_sq_values = _safe_sparse_mask(exp_avg_sq, grad)._values()
                    exp_avg_sq_update_values = grad_values.pow(2).sub_(old_exp_avg_sq_values).mul_(1 - beta2)
                    exp_avg_sq.add_(make_sparse(exp_avg_sq_update_values))

                    # Dense addition again is intended, avoiding another sparse_mask
                    numer = exp_avg_update_values.add_(old_exp_avg_values)
                    exp_avg_sq_update_values.add_(old_exp_avg_sq_values)
                    denom = exp_avg_sq_update_values.sqrt_().add_(group['eps'])
                    del exp_avg_update_values, exp_avg_sq_update_values

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                    p.data.add_(make_sparse(-step_size * numer.div_(denom)))

                else:
                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                    p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
