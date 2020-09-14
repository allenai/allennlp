from typing import Optional, Dict

import torch


class TransformerModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def _get_mapped_submodules(cls, pretrained_module, source="huggingface"):
        """
        Subclasses overload this method, and provide appropriate name mapping based on the source.
        """
        return dict(pretrained_module.named_modules())

    def _construct_default_mapping(self, source):
        """
        Recursively constructs the default mapping of parameter names for loading pretrained module weights.
        Keys are from expected pretrained module, and values are parameter names from this module.
        """
        mapping = {}
        for name, module in self.named_modules():
            if name != "":
                if hasattr(module, "_construct_default_mapping"):
                    module._construct_default_mapping(source)
                    mapping.update(
                        module._default_mapping
                    )  # FIX: can potentially cause collisions.
        self._default_mapping = mapping

    def _load_from_pretrained_module(
        self,
        pretrained_module: torch.nn.Module,
        source="huggingface",
        mapping: Optional[Dict[str, str]] = None,
    ):
        """
        Loads the weights of the `pretrained_module` into the instance.
        Optionally, a `mapping` is specified for any differences in parameter names
        between `pretrained_module` and the instance.
        """
        if mapping is None:
            self._construct_default_mapping(source)
            mapping = self._default_mapping
        module_parameters = dict(self.named_parameters())
        for name, parameter in pretrained_module.named_parameters():
            for key, val in mapping.items():
                # so that we replace the names of submodules too.
                # eg. module.key.anothermodule --> module.val.anothermodule
                name = name.replace(key, val)
            if name not in module_parameters:
                raise ValueError(
                    f"Couldn't find a matching parameter for {name}. Is this module "
                    "compatible with the pretrained module you're using?"
                )
            module_parameters[name].data.copy_(parameter.data)
