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
import copy
import logging
import re
import math
from typing import Any, Dict, List, Tuple, Union, Optional

from overrides import overrides
import torch
import transformers

from allennlp.common import Params, Registrable, Lazy
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)


ParameterGroupsType = List[Tuple[List[str], Dict[str, Any]]]


def make_parameter_groups(
    model_parameters: List[Tuple[str, torch.nn.Parameter]],
    groups: Optional[ParameterGroupsType] = None,
) -> Union[List[Dict[str, Any]], List[torch.nn.Parameter]]:
    """
    Takes a list of model parameters with associated names (typically coming from something like
    `model.named_parameters()`), along with a grouping (as specified below), and prepares them to be passed
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

    All of key-value pairs specified in each of these dictionaries will passed passed as-is
    to the optimizer, with the exception of a dictionaries that specify `requires_grad` to be `False`:

    ```
    [
        ...
        (["regex"], {"requires_grad": False})
    ]
    ```

    When a parameter group has `{"requires_grad": False}`, the gradient on all matching parameters
    will be disabled and that group will be dropped so that it's not actually passed to the optimizer.

    Ultimately, the return value of this function is in the right format to be passed directly
    as the `params` argument to a pytorch `Optimizer`.
    If there are multiple groups specified, this is a list of dictionaries, where each
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

        # find and remove any groups with 'requires_grad = False'
        no_grad_group_indices: List[int] = []
        for k, (names, group) in enumerate(zip(parameter_group_names, parameter_groups)):
            if group.get("requires_grad") is False:
                no_grad_group_indices.append(k)
                logger.info("Disabling gradient for the following parameters: %s", names)
                for param in group["params"]:
                    param.requires_grad_(False)

                # warn about any other unused options in that group.
                unused_options = {
                    key: val for key, val in group.items() if key not in ("params", "requires_grad")
                }
                if unused_options:
                    logger.warning("Ignoring unused options %s for %s", unused_options, names)
        parameter_group_names = [
            names
            for (k, names) in enumerate(parameter_group_names)
            if k not in no_grad_group_indices
        ]
        parameter_groups = [
            group for (k, group) in enumerate(parameter_groups) if k not in no_grad_group_indices
        ]

        # log the remaining parameter groups
        logger.info("Done constructing parameter groups.")
        for k in range(len(parameter_groups)):
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


class Optimizer(torch.optim.Optimizer, Registrable):
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


@Optimizer.register("multi")
class MultiOptimizer(Optimizer):
    """
    A `MultiOptimizer` creates a dictionary of `Optimizer`s keyed on some 'name'.
    Each Optimizer contains its own set of parameters which are obtained using
    regex matches for certain model parameters.

    This optimizer works by taking in a parameter `optimizers` which contains a list of `Optimizers`
    with their keyword arguments, and a parameter `parameter_groups`, which contains regexes and their
    corresponding optimizer and optional non-default optimizer options for this group.
    The regexes in the parameter groups are assigned to their optimizer based on the 'name' argument
    where the 'name' value should be the same for the optimizer and parameter group.
    You should specify a default optimizer with 'name': 'default' which will be used for all
    parameters which didn't obtain a regex match or when your parameter group doesn't contain a 'name'
    parameter.

    # Parameters

    optimizers: `List[Dict[str, Any]]`
        A list of optimizers to use. Each entry in the list is a dictionary of keyword arguments. A 'name'
        keyword argument should be given which will serve as the key to match the optimizer with a
        specific parameter group. You should also supply an entry for the default parameter group,
        e.g. 'name': 'default'.

    parameter_groups:  `List[Tuple[List[str], Dict[str, Any]]`, optional (default = `None`)
        See the docstring of `make_parameter_groups` for what this parameter should look like. It
        should follow the same format as there, except an additional 'optimizer_name' argument should be
        provided to match this group to its own optimizer. Optimizer options can also be set for this
        group which will override the default options.
    """

    def __init__(
        self,
        model_parameters: List[Tuple[str, torch.nn.Parameter]],
        optimizers: Dict[str, Lazy[Optimizer]],
        parameter_groups: ParameterGroupsType,
    ):
        if "default" not in optimizers:
            raise ConfigurationError(
                "No optimizer was provided for the 'default' group."
                " Please provide an Optimizer under the name 'default'"
            )

        # calculate parameter groups for each optimizer
        optimizer_name_to_parameter_groups: Dict[str, ParameterGroupsType] = {
            optimizer_name: [] for optimizer_name in optimizers.keys()
        }
        for parameter_group in parameter_groups:
            regexes, pg_overrides = parameter_group
            optimizer_name = pg_overrides.get("optimizer_name", "default")
            optimizer_name_to_parameter_groups[optimizer_name].append(parameter_group)

        # calculate model parameters for each optimizer
        optimizer_name_to_model_parameters: Dict[str, List[Tuple[str, torch.nn.Parameter]]] = {
            optimizer_name: [] for optimizer_name in optimizers.keys()
        }
        for model_parameter_tuple in model_parameters:
            parameter_name, parameter_tensor = model_parameter_tuple
            for regexes, pg_overrides in parameter_groups:
                if any(re.search(regex, parameter_name) for regex in regexes):
                    optimizer_name = pg_overrides.get("optimizer_name", "default")
                    optimizer_name_to_model_parameters[optimizer_name].append(model_parameter_tuple)
                    break
            else:
                optimizer_name_to_model_parameters["default"].append(model_parameter_tuple)

        # Check to see if an optimizer didn't receive any parameters.
        for optimizer_name, optimizer_parameters in optimizer_name_to_model_parameters.items():
            if optimizer_name != "default" and len(optimizer_parameters) == 0:
                raise ConfigurationError(
                    f"Optimizer '{optimizer_name}' did not receive any parameters."
                    " If you are using `parameter_groups`, please make sure that the regexes you have provided"
                    " match the desired model parameters, or that the `name` value of this optimizer "
                    " matches that of the parameter group you are trying to assign to it."
                    " Alternatively, you can remove this optimizer from the provided `optimizers`"
                    " if it is not relevant to a particular parameter group."
                )
        # default optimizer is required, but associating it with model parameters is optional
        if len(optimizer_name_to_model_parameters["default"]) == 0:
            del optimizers["default"]
            del optimizer_name_to_model_parameters["default"]
            del optimizer_name_to_parameter_groups["default"]

        self.optimizers = {
            optimizer_name: lazy_optimizer.construct(
                model_parameters=optimizer_name_to_model_parameters[optimizer_name],
                parameter_groups=optimizer_name_to_parameter_groups[optimizer_name],
            )
            for optimizer_name, lazy_optimizer in optimizers.items()
        }

        # Copy the defaults from the optimizers into the parameter groups, so they are visible in
        # optimizer.params when optimizer is a MultiOptimizer.
        parameter_groups = copy.deepcopy(parameter_groups)
        for parameter_group in parameter_groups:
            regexes, pg_overrides = parameter_group
            optimizer_name = pg_overrides.get("optimizer_name", "default")
            optimizer = self.optimizers[optimizer_name]
            for key, value in optimizer.defaults.items():
                if key not in pg_overrides:
                    pg_overrides[key] = value

        # Any parameter that doesn't match a regex goes into the last output from make_parameter_groups(). We need
        # to copy the params from the default optimizer into that groups as well.
        made_parameter_groups = make_parameter_groups(model_parameters, parameter_groups)
        if "default" in self.optimizers:
            for key, value in self.optimizers["default"].defaults.items():
                made_parameter_groups[-1][key] = value

        super().__init__(made_parameter_groups, {})

    @overrides
    def step(self):
        """
        Takes an optimization step for each optimizer.
        """
        for optimizer in self.optimizers.values():
            optimizer.step()

    @overrides
    def state_dict(self):
        """
        Creates an object `optimizer_state_dict`, which is a dictionary mapping an optimizer key to its
        `state_dict`. This dictionary is used as the value for 'optimizer' in the 'training_states' dictionary in
        the `gradient_descent` `Trainer`, e.g.
        ```
        "optimizer" : {
            "optimizer1": `optimizer1_state_dict`,
            "optimizer2": `optimizer2_state_dict`
        }.
        ```
        """
        optimizer_state_dict = {
            f"{optimizer_key}_optimizer": optimizer.state_dict()
            for optimizer_key, optimizer in self.optimizers.items()
        }

        return optimizer_state_dict

    @overrides
    def load_state_dict(self, training_state: Dict[str, Any]):
        """
        Loads each optimizer's `state_dict`.
        """
        for optimizer_key, optimizer in self.optimizers.items():
            optimizer.load_state_dict(training_state[f"{optimizer_key}_optimizer"])

    @overrides
    def zero_grad(self, set_to_none: bool = False):
        """
        Sets parameter gradients to zero or None.
        """
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none)


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
        lr: float = 1e-5,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
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
