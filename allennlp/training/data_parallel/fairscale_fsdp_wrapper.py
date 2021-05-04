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

    In a typical AllenNLP configuration file, `model`, `cuda_device`, and `mixed_precision`
    should not be specified. `model` and `cuda_device` are self-explanatory. `mixed_precision`
    will correspond to the value of `use_amp` in the `GradientDescentTrainer`.
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
        if self.cuda_device != torch.device("cpu"):
            self._wrapped_model = self._wrapped_model.cuda()

    @overrides
    def get_wrapped_model(self) -> torch.nn.Module:
        return self._wrapped_model
