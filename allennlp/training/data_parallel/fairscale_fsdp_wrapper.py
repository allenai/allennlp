from typing import Union

from fairscale.nn import FullyShardedDataParallel as FSDP
from overrides import overrides
import torch

from allennlp.training.data_parallel.ddp_wrapper import DdpWrapper
from allennlp.models import Model


@DdpWrapper.register("fairscale_fsdp")
class FairScaleFsdpWrapper(DdpWrapper):
    """
    A `DdpWrapper` for FairScale's [`FullyShardedDataParallel`]
    (https://fairscale.readthedocs.io/en/latest/api/nn/fsdp.html).
    """

    def __init__(
        self,
        model: Model,
        cuda_device: Union[torch.device, int] = -1,
        mixed_precision: bool = False,
        reshard_after_forward: bool = True,
        cpu_offload: bool = False,
        flatten_parameters: bool = True,
    ) -> None:
        super().__init__(model, cuda_device=cuda_device)
        self._wrapped_model = FSDP(
            self.model,
            compute_device=self.cuda_device,
            mixed_precision=mixed_precision,
            reshard_after_forward=reshard_after_forward,
            cpu_offload=cpu_offload,
            flatten_parameters=flatten_parameters,
        )

    @overrides
    def get_wrapped_model(self) -> torch.nn.Module:
        return self._wrapped_model
