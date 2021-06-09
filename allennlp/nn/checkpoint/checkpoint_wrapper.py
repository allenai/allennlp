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

    # Need a dummy tensor with `requires_grad=True` or all gradients will be None,
    # so we have to wrap the `original_forward` method with a function that takes the
    # dummy tensor (it doesn't have to do anythning with it).

    def run_function(dummy_tensor, *forward_args, **forward_kwargs):
        return original_forward(module, *forward_args, **forward_kwargs)

    dummy_tensor = torch.tensor([], requires_grad=True)
    return CheckpointFunction.apply(run_function, True, dummy_tensor, *args, **kwargs)
