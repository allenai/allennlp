import functools
from typing import Optional, Callable, TypeVar
import weakref

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential
from overrides import overrides

from allennlp.common.registrable import Registrable


class CheckpointWrapper(Registrable):
    """
    A `CheckpointWrapper` is used to enable activation/gradient checkpointing on modules
    that you wrap via the `.wrap_module()` method.

    !!! Note
        Some implementations have restrictions on the type of `nn.Module` you can wrap,
        and the inputs that can be passed to the wrapped module.
        For instance, the default implementation `TorchCheckpointWrapper` only accepts
        `nn.Sequential` modules that take a single tensor input.

    !!! Note
        The provided `CheckpointWrapper` implementations work by patching the `.forward()` method
        of the module, so there is no easy way to disable checkpointing
        in a module after `.wrap_module()` is called.
    """

    default_implementation = "torch"

    def wrap_module(self, module: nn.Module, **kwargs) -> nn.Module:
        raise NotImplementedError


@CheckpointWrapper.register("torch")
class TorchCheckpointWrapper(CheckpointWrapper):
    @overrides
    def wrap_module(self, module: nn.Sequential, segments: Optional[int] = None) -> nn.Sequential:
        """
        Wrap a `nn.Sequential` module for activation checkpointing.
        """
        if not isinstance(module, nn.Sequential):
            raise ValueError(
                f"TorchCheckpointWrapper can only wrap nn.Sequential modules, but got {module.__class__}"
            )
        if segments is None:
            segments = len(module)
        # Adapted from FairScale:
        #  --> https://github.com/facebookresearch/fairscale/blob/1e4a503cda8571851a68effd6e504a192838ab06/fairscale/nn/checkpoint/checkpoint_activations.py#L145-L153  # noqa: E501
        # We just patch the forward method to avoid having to proxy all the fields and other methods.
        # The use of weakref here is to prevent creating a ref cycle: m -> m.forward -> m.
        module.forward = functools.partial(  # type: ignore[assignment]
            _checkpointed_forward, type(module).forward, weakref.ref(module), segments
        )
        return module


_T = TypeVar("_T")


def _checkpointed_forward(
    original_forward: Callable[[nn.Sequential, torch.Tensor], _T],
    weak_self,
    segments: int,
    inputs: torch.Tensor,
) -> _T:
    module = weak_self()
    assert (
        module is not None
    ), "patched forward method called after module has been garbage collected!"

    # If in eval mode, just use the original `.forward()` method.
    if not module.training:
        return original_forward(module, inputs)

    # Annoyingly the input needs a gradient or all gradients will be None.
    if not inputs.requires_grad:
        inputs.requires_grad_()
    return checkpoint_sequential(module, segments, inputs)
