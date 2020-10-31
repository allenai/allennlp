from typing import List, Tuple, Dict, Any

import torch

from apex.optimizers.fused_adam import FusedAdam
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.lamb import FusedLamb
from deepspeed.runtime.fp16.onebit_adam import OnebitAdam

from allennlp.training.optimizers import Optimizer, make_parameter_groups

@Optimizer.register("fused_adam")
class FusedAdamOptimizer(Optimizer, FusedAdam):
    def __init__(
        self,
        model_parameters: List[Tuple[str, torch.nn.Parameter]],
        parameter_groups: List[Tuple[List[str], Dict[str, Any]]] = None,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        bias_correction: bool =True,
        adam_w_mode: bool = True,
        set_grad_none: bool = True,
    ):
        super().__init__(
            params=make_parameter_groups(model_parameters, parameter_groups),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            bias_correction=bias_correction,
            adam_w_mode=adam_w_mode,
            set_grad_none=set_grad_none,
        )

# This does not currently work
@Optimizer.register("cpu_adam")
class DeepspeedCPUAdamOptimizer(Optimizer, DeepSpeedCPUAdam):
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
            model_params=make_parameter_groups(model_parameters, parameter_groups),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )

@Optimizer.register("fused_lamb")
class FusedLambOptimizer(Optimizer, FusedLamb):
    def __init__(
        self,
        model_parameters: List[Tuple[str, torch.nn.Parameter]],
        parameter_groups: List[Tuple[List[str], Dict[str, Any]]] = None,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        eps_inside_sqrt: bool = False,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        max_grad_norm: float = 0.,
        max_coeff: float = 10.0,
        min_coeff: float = 0.01
    ):
        super().__init__(
            params=make_parameter_groups(model_parameters, parameter_groups),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            max_grad_norm=max_grad_norm,
            max_coeff=max_coeff,
            min_coeff=min_coeff,
        )