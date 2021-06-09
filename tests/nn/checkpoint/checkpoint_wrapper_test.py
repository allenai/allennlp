import torch
import torch.nn as nn

from allennlp.common.params import Params
from allennlp.common.testing import multi_device
from allennlp.nn.checkpoint.checkpoint_wrapper import CheckpointWrapper, TorchCheckpointWrapper


forward_calls = 0


class LinearForTesting(nn.Linear):
    def forward(self, *args, **kwargs):
        global forward_calls
        forward_calls += 1
        return super().forward(*args, **kwargs)


def setup_function(_):
    global forward_calls
    forward_calls = 0


class ModuleForTesting(nn.Module):
    def __init__(self, checkpoint_wrapper: CheckpointWrapper) -> None:
        super().__init__()
        ffn = nn.Sequential(
            nn.Linear(3, 3),
            # Use a nn.Dropout layer to test RNG (random number generator) state save/restore
            nn.Dropout(p=0.5),
            LinearForTesting(3, 3),
            nn.Linear(3, 3),
        )
        self.ffn = checkpoint_wrapper.wrap_module(ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x).sum(-1)


@multi_device
def test_torch_checkpoint_wrapper(device: str):
    device_ = torch.device(device)

    checkpoint_wrapper: TorchCheckpointWrapper = CheckpointWrapper.from_params(  # type: ignore[assignment]
        Params({"type": "torch"})
    )
    module = ModuleForTesting(checkpoint_wrapper).to(device_)
    optim = torch.optim.Adam(module.parameters(), lr=0.0001)

    # Test forward pass
    module.train()
    # Shape (batch_size = 2, 3)
    x = torch.randn(2, 3).to(device_)
    loss = module(x).sum()
    assert forward_calls == 1, f"wrong number of forward calls: {forward_calls}"

    # Test backward pass + optimizer step.
    loss.backward()
    assert forward_calls == 2, f"wrong number of forward calls: {forward_calls}"
    for param in module.parameters():
        assert param.grad is not None
    optim.step()
    optim.zero_grad(set_to_none=True)

    # Now test forward pass in eval mode.
    module.eval()
    x = torch.randn(2, 3).to(device_)
    module(x)
