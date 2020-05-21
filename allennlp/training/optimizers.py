"""
AllenNLP just uses
[PyTorch optimizers](https://pytorch.org/docs/master/optim.html),
with a thin wrapper to allow registering them and instantiating them `from_params`.

The available optimizers are

* [adadelta](https://pytorch.org/docs/master/optim.html#torch.optim.Adadelta)
* [adagrad](https://pytorch.org/docs/master/optim.html#torch.optim.Adagrad)
* [adam](https://pytorch.org/docs/master/optim.html#torch.optim.Adam)
* [adamw](https://pytorch.org/docs/master/optim.html#torch.optim.AdamW)
* [huggingface_adamw](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.AdamW)
* [sparse_adam](https://pytorch.org/docs/master/optim.html#torch.optim.SparseAdam)
* [sgd](https://pytorch.org/docs/master/optim.html#torch.optim.SGD)
* [rmsprop](https://pytorch.org/docs/master/optim.html#torch.optim.RMSprop)
* [adamax](https://pytorch.org/docs/master/optim.html#torch.optim.Adamax)
* [averaged_sgd](https://pytorch.org/docs/master/optim.html#torch.optim.ASGD)
"""

import logging
import re
import math
from typing import Any, Dict, List, Tuple, Union

import torch
import transformers

from allennlp.common import Params, Registrable

logger = logging.getLogger(__name__)


def make_parameter_groups(
    model_parameters: List[Tuple[str, torch.nn.Parameter]],
    groups: List[Tuple[List[str], Dict[str, Any]]] = None,
) -> Union[List[Dict[str, Any]], List[torch.nn.Parameter]]:
    """
    Takes a list of model parameters with associated names (typically coming from something like
    `model.parameters`), along with a grouping (as specified below), and prepares them to be passed
    to the `__init__` function of a `torch.Optimizer`.  This means separating the parameters into
    groups with the given regexes, and prepping whatever keyword arguments are given for those
    regexes in `groups`.

    `groups` contains something like:

    ```
    [
        (["regex1", "regex2"], {"lr": 1e-3}),
        (["regex3"], {"lr": 1e-4})
    ]
    ```

    The return value in the right format to be passed directly as the `params` argument to a pytorch
    `Optimizer`.  If there are multiple groups specified, this is list of dictionaries, where each
    dict contains a "parameter group" and groups specific options, e.g., {'params': [list of
    parameters], 'lr': 1e-3, ...}.  Any config option not specified in the additional options (e.g.
    for the default group) is inherited from the top level arguments given in the constructor.  See:
    <https://pytorch.org/docs/0.3.0/optim.html?#per-parameter-options>.  See also our
    `test_optimizer_parameter_groups` test for an example of how this works in this code.

    The dictionary's return type is labeled as `Any`, because it can be a `List[torch.nn.Parameter]`
    (for the "params" key), or anything else (typically a float) for the other keys.
    """
    if groups:
        # In addition to any parameters that match group specific regex,
        # we also need a group for the remaining "default" group.
        # Those will be included in the last entry of parameter_groups.
        parameter_groups: Union[List[Dict[str, Any]], List[torch.nn.Parameter]] = [
            {"params": []} for _ in range(len(groups) + 1)
        ]
        # add the group specific kwargs
        for k in range(len(groups)):
            parameter_groups[k].update(groups[k][1])

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
                            raise ValueError(
                                "{} was specified in two separate parameter groups".format(name)
                            )
                        group_index = k
                        regex_use_counts[regex] += 1

            if group_index is not None:
                parameter_groups[group_index]["params"].append(param)
                parameter_group_names[group_index].add(name)
            else:
                # the default group
                parameter_groups[-1]["params"].append(param)
                parameter_group_names[-1].add(name)

        # log the parameter groups
        logger.info("Done constructing parameter groups.")
        for k in range(len(groups) + 1):
            group_options = {
                key: val for key, val in parameter_groups[k].items() if key != "params"
            }
            logger.info("Group %s: %s, %s", k, list(parameter_group_names[k]), group_options)
        # check for unused regex
        for regex, count in regex_use_counts.items():
            if count == 0:
                logger.warning(
                    "When constructing parameter groups, %s does not match any parameter name",
                    regex,
                )

    else:
        parameter_groups = [param for name, param in model_parameters]

    # Log the number of parameters to optimize
    num_parameters = 0
    for parameter_group in parameter_groups:
        if isinstance(parameter_group, dict):
            num_parameters += sum(parameter.numel() for parameter in parameter_group["params"])
        else:
            num_parameters += parameter_group.numel()  # type: ignore
    logger.info("Number of trainable parameters: %s", num_parameters)
    return parameter_groups


class Optimizer(Registrable):
    """
    This class just allows us to implement `Registrable` for Pytorch Optimizers.  We do something a
    little bit different with `Optimizers`, because they are implemented as classes in PyTorch, and
    we want to use those classes.  To make things easy, we just inherit from those classes, using
    multiple inheritance to also inherit from `Optimizer`.  The only reason we do this is to make
    type inference on parameters possible, so we can construct these objects using our configuration
    framework.  If you are writing your own script, you can safely ignore these classes and just use
    the `torch.optim` classes directly.

    If you are implementing one of these classes, the `model_parameters` and `parameter_groups`
    arguments to `__init__` are important, and should always be present.  The trainer will pass
    the trainable parameters in the model to the optimizer using the name `model_parameters`, so if
    you use a different name, your code will crash.  Nothing will technically crash if you use a
    name other than `parameter_groups` for your second argument, it will just be annoyingly
    inconsistent.

    Most subclasses of `Optimizer` take both a `model_parameters` and a `parameter_groups`
    constructor argument.  The `model_parameters` argument does not get an entry in a typical
    AllenNLP configuration file, but the `parameter_groups` argument does (if you want a non-default
    value).  See the documentation for the `make_parameter_groups` function for more information on
    how the `parameter_groups` argument should be specified.
    """

    default_implementation = "adam"

    @staticmethod
    def default(model_parameters: List) -> "Optimizer":
        return Optimizer.from_params(model_parameters=model_parameters, params=Params({}))


@Optimizer.register("adam")
class AdamOptimizer(Optimizer, torch.optim.Adam):
    """
    Registered as an `Optimizer` with name "adam".
    """

    def __init__(
        self,
        model_parameters: List[Tuple[str, torch.nn.Parameter]],
        parameter_groups: List[Tuple[List[str], Dict[str, Any]]] = None,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        super().__init__(
            params=make_parameter_groups(model_parameters, parameter_groups),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )


@Optimizer.register("sparse_adam")
class SparseAdamOptimizer(Optimizer, torch.optim.SparseAdam):
    """
    Registered as an `Optimizer` with name "sparse_adam".
    """

    def __init__(
        self,
        model_parameters: List[Tuple[str, torch.nn.Parameter]],
        parameter_groups: List[Tuple[List[str], Dict[str, Any]]] = None,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
    ):
        super().__init__(
            params=make_parameter_groups(model_parameters, parameter_groups),
            lr=lr,
            betas=betas,
            eps=eps,
        )


@Optimizer.register("adamax")
class AdamaxOptimizer(Optimizer, torch.optim.Adamax):
    """
    Registered as an `Optimizer` with name "adamax".
    """

    def __init__(
        self,
        model_parameters: List[Tuple[str, torch.nn.Parameter]],
        parameter_groups: List[Tuple[List[str], Dict[str, Any]]] = None,
        lr: float = 0.002,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ):
        super().__init__(
            params=make_parameter_groups(model_parameters, parameter_groups),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )


@Optimizer.register("adamw")
class AdamWOptimizer(Optimizer, torch.optim.AdamW):
    """
    Registered as an `Optimizer` with name "adamw".
    """

    def __init__(
        self,
        model_parameters: List[Tuple[str, torch.nn.Parameter]],
        parameter_groups: List[Tuple[List[str], Dict[str, Any]]] = None,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
    ):
        super().__init__(
            params=make_parameter_groups(model_parameters, parameter_groups),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )


@Optimizer.register("huggingface_adamw")
class HuggingfaceAdamWOptimizer(Optimizer, transformers.AdamW):
    """
    Registered as an `Optimizer` with name "huggingface_adamw".
    """

    def __init__(
        self,
        model_parameters: List[Tuple[str, torch.nn.Parameter]],
        parameter_groups: List[Tuple[List[str], Dict[str, Any]]] = None,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-06,
        weight_decay: float = 0.0,
        correct_bias: bool = False,
    ):
        super().__init__(
            params=make_parameter_groups(model_parameters, parameter_groups),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
        )


@Optimizer.register("adagrad")
class AdagradOptimizer(Optimizer, torch.optim.Adagrad):
    """
    Registered as an `Optimizer` with name "adagrad".
    """

    def __init__(
        self,
        model_parameters: List[Tuple[str, torch.nn.Parameter]],
        parameter_groups: List[Tuple[List[str], Dict[str, Any]]] = None,
        lr: float = 0.01,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1e-10,
    ):
        super().__init__(
            params=make_parameter_groups(model_parameters, parameter_groups),
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            eps=eps,
        )


@Optimizer.register("adadelta")
class AdadeltaOptimizer(Optimizer, torch.optim.Adadelta):
    """
    Registered as an `Optimizer` with name "adadelta".
    """

    def __init__(
        self,
        model_parameters: List[Tuple[str, torch.nn.Parameter]],
        parameter_groups: List[Tuple[List[str], Dict[str, Any]]] = None,
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-06,
        weight_decay: float = 0.0,
    ):
        super().__init__(
            params=make_parameter_groups(model_parameters, parameter_groups),
            lr=lr,
            rho=rho,
            eps=eps,
            weight_decay=weight_decay,
        )


@Optimizer.register("sgd")
class SgdOptimizer(Optimizer, torch.optim.SGD):
    """
    Registered as an `Optimizer` with name "sgd".
    """

    def __init__(
        self,
        model_parameters: List[Tuple[str, torch.nn.Parameter]],
        lr: float,
        parameter_groups: List[Tuple[List[str], Dict[str, Any]]] = None,
        momentum: float = 0.0,
        dampening: float = 0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        super().__init__(
            params=make_parameter_groups(model_parameters, parameter_groups),
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )


@Optimizer.register("rmsprop")
class RmsPropOptimizer(Optimizer, torch.optim.RMSprop):
    """
    Registered as an `Optimizer` with name "rmsprop".
    """

    def __init__(
        self,
        model_parameters: List[Tuple[str, torch.nn.Parameter]],
        parameter_groups: List[Tuple[List[str], Dict[str, Any]]] = None,
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
    ):
        super().__init__(
            params=make_parameter_groups(model_parameters, parameter_groups),
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
        )


@Optimizer.register("averaged_sgd")
class AveragedSgdOptimizer(Optimizer, torch.optim.ASGD):
    """
    Registered as an `Optimizer` with name "averaged_sgd".
    """

    def __init__(
        self,
        model_parameters: List[Tuple[str, torch.nn.Parameter]],
        parameter_groups: List[Tuple[List[str], Dict[str, Any]]] = None,
        lr: float = 0.01,
        lambd: float = 0.0001,
        alpha: float = 0.75,
        t0: float = 1000000.0,
        weight_decay: float = 0.0,
    ):
        super().__init__(
            params=make_parameter_groups(model_parameters, parameter_groups),
            lr=lr,
            lambd=lambd,
            alpha=alpha,
            t0=t0,
            weight_decay=weight_decay,
        )


@Optimizer.register("dense_sparse_adam")
class DenseSparseAdam(Optimizer, torch.optim.Optimizer):
    """
    NOTE: This class has been copied verbatim from the separate Dense and
    Sparse versions of Adam in Pytorch.

    Implements Adam algorithm with dense & sparse gradients.
    It has been proposed in Adam: A Method for Stochastic Optimization.

    Registered as an `Optimizer` with name "dense_sparse_adam".

    # Parameters

    params : `iterable`
        iterable of parameters to optimize or dicts defining parameter groups
    lr : `float`, optional (default = `1e-3`)
        The learning rate.
    betas : `Tuple[float, float]`, optional (default = `(0.9, 0.999)`)
        coefficients used for computing running averages of gradient
        and its square.
    eps : `float`, optional, (default = `1e-8`)
        A term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        model_parameters: List[Tuple[str, torch.nn.Parameter]],
        parameter_groups: List[Tuple[List[str], Dict[str, Any]]] = None,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(make_parameter_groups(model_parameters, parameter_groups), defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        # Parameters

        closure : `callable`, optional.
            A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                state["step"] += 1

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

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
                    old_exp_avg_values = exp_avg.sparse_mask(grad)._values()
                    exp_avg_update_values = grad_values.sub(old_exp_avg_values).mul_(1 - beta1)
                    exp_avg.add_(make_sparse(exp_avg_update_values))
                    old_exp_avg_sq_values = exp_avg_sq.sparse_mask(grad)._values()
                    exp_avg_sq_update_values = (
                        grad_values.pow(2).sub_(old_exp_avg_sq_values).mul_(1 - beta2)
                    )
                    exp_avg_sq.add_(make_sparse(exp_avg_sq_update_values))

                    # Dense addition again is intended, avoiding another sparse_mask
                    numer = exp_avg_update_values.add_(old_exp_avg_values)
                    exp_avg_sq_update_values.add_(old_exp_avg_sq_values)
                    denom = exp_avg_sq_update_values.sqrt_().add_(group["eps"])
                    del exp_avg_update_values, exp_avg_sq_update_values

                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                    p.data.add_(make_sparse(-step_size * numer.div_(denom)))

                else:
                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
