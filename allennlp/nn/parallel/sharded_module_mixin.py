import torch


class ShardedModuleMixin:
    """
    Mixin class for sharded data parallel wrappers. Subclasses should implement
    `get_original_module()` which returns a reference the original inner wrapped module.
    """

    def get_original_module(self) -> torch.nn.Module:
        """
        Get the original
        """
        raise NotImplementedError
