from typing import Optional

import torch
import torch.nn as nn

from allennlp.common.params import Params
from allennlp.common.testing import multi_device
from allennlp.nn.checkpoint.checkpoint_wrapper import CheckpointWrapper, TorchCheckpointWrapper


class ModuleForTesting(nn.Module):
    def __init__(self, checkpoint_wrapper: Optional[CheckpointWrapper] = None) -> None:
        super().__init__()
        ffn = nn.Sequential(
            nn.Linear(3, 3),
            # Use a nn.Dropout layer to test RNG (random number generator) state save/restore
            nn.Dropout(p=0.5),
            nn.Linear(3, 3),
        )
        if checkpoint_wrapper is not None:
            self.ffn = checkpoint_wrapper.wrap_module(ffn)
        else:
            self.ffn = ffn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x).sum(-1)


@multi_device
def test_torch_checkpointer_wrapper_forward_backward(device: str):
    device_ = torch.device(device)

    checkpointer_wrapper: TorchCheckpointWrapper = CheckpointWrapper.from_params(  # type: ignore[assignment]
        Params({"type": "torch"})
    )
    module = ModuleForTesting(checkpointer_wrapper).to(device_)
    optim = torch.optim.Adam(module.parameters(), lr=0.0001)

    # Test forward pass
    module.train()
    # Shape (batch_size = 2, 3)
    x = torch.randn(2, 3).to(device_)
    loss = module(x).sum()

    # Test backward pass + optimizer step.
    loss.backward()
    assert module.ffn[0].weight.grad is not None  # type: ignore[index]
    optim.step()
    optim.zero_grad(set_to_none=True)

    # Now test forward pass in eval mode.
    module.eval()
    x = torch.randn(2, 3).to(device_)
    module(x)
