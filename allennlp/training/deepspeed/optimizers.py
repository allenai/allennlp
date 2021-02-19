from typing import List, Tuple, Dict, Any

import torch

from deepspeed.ops.lamb import FusedLamb
from allennlp.training.optimizers import Optimizer, make_parameter_groups


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
        max_grad_norm: float = 0.0,
        max_coeff: float = 10.0,
        min_coeff: float = 0.01,
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
