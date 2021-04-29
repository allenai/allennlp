from typing import List, Dict, Any, Tuple

import fairscale
import torch

from allennlp.common.lazy import Lazy
from allennlp.training.optimizers.optimizer import Optimizer, make_parameter_groups


@Optimizer.register("fairscale_oss")
class FairScaleOssOptimizer(fairscale.optim.OSS, Optimizer):
    """
    FairScale's OSS optimizer.
    """

    def __init__(
        self,
        model_parameters: List[Tuple[str, torch.nn.Parameter]],
        optimizer: Lazy[Optimizer],
        parameter_groups: List[Tuple[List[str], Dict[str, Any]]] = None,
        broadcast_fp16: bool = False,
    ) -> None:
        super().__init__(
            make_parameter_groups(model_parameters),
            optim=self._optimizer_constructor,
            broadcast_fp16=broadcast_fp16,
        )
        self._lazy_optimizer = optimizer

    def _optimizer_constructor(self, params, **defaults) -> Optimizer:
        return self._lazy_optimizer.construct(model_parameters=params, **defaults)
