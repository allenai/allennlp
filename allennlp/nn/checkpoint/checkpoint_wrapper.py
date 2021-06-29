import functools
from typing import Callable, TypeVar
import weakref

import torch
import torch.nn as nn
from torch.utils.checkpoint import CheckpointFunction
from overrides import overrides

from allennlp.common.registrable import Registrable


class CheckpointWrapper(Registrable):
    """
    A `CheckpointWrapper` is used to enable activation/gradient checkpointing on modules
    that you wrap via the `.wrap_module()` method.
    """

    default_implementation = "torch"

    def wrap_module(self, module: nn.Module, **kwargs) -> nn.Module:
        raise NotImplementedError


@CheckpointWrapper.register("torch")
class TorchCheckpointWrapper(CheckpointWrapper):
    @overrides
    def wrap_module(self, module: nn.Module) -> nn.Module:
        """
        Wrap a module so that the forward method uses PyTorch's [checkpointing functionality]
        (https://pytorch.org/docs/stable/checkpoint.html).

        !!! Note
            Currently this `CheckpointWrapper` implementation requires that the wrapped
            module is called with positional arguments only.

            We recommend you use the
            :class:`allennlp.nn.checkpoint.fairscale_checkpoint_wrapper.FairScaleCheckpointWrapper`
            if you need more flexibility.

        """
        # Inspired by FairScale:
        #  --> https://github.com/facebookresearch/fairscale/blob/1e4a503cda8571851a68effd6e504a192838ab06/fairscale/nn/checkpoint/checkpoint_activations.py#L145-L153  # noqa: E501
        # We just patch the forward method to avoid having to proxy all the fields and other methods.
        # The use of weakref here is to prevent creating a ref cycle: m -> m.forward -> m.
        module.forward = functools.partial(  # type: ignore[assignment]
            _checkpointed_forward, type(module).forward, weakref.ref(module)
        )
        return module


_T = TypeVar("_T")


def _checkpointed_forward(
    original_forward: Callable[..., _T],
    weak_self,
    *args,
    **kwargs,
) -> _T:
    module = weak_self()
    assert (
        module is not None
    ), "patched forward method called after module has been garbage collected!"

    # If in eval mode, just use the original `.forward()` method.
    if not module.training:
        return original_forward(module, *args, **kwargs)

    # The function that the `CheckpointFunction` is applied to needs to have at least one
    # input tensor that has `requires_grad=True`.
    # So to avoid having to make users manually set the `requires_grad` flag on input tensors,
    # we wrap the `original_forward` with a `run_function` that takes an additional dummy tensor
    # as input, and we'll set `requires_grad=True` on this dummy tensor.

    def run_function(dummy_tensor, *forward_args, **forward_kwargs):
        return original_forward(module, *forward_args, **forward_kwargs)

    dummy_tensor = torch.tensor([], requires_grad=True)
    return CheckpointFunction.apply(run_function, True, dummy_tensor, *args, **kwargs)
