from typing import Optional, Dict

import torch


class TransformerModule(torch.nn.Module):
    """
    Base class to help with generalized loading of pretrained weights.
    """

    _huggingface_mapping: Dict[str, str] = {}

    def __init__(self, *args, **kwargs):
        super().__init__()

    @classmethod
    def _get_mapped_submodules(
        cls, pretrained_module, source="huggingface", mapping: Optional[Dict[str, str]] = None
    ):
        """
        Subclasses overload this method, and provide appropriate name mapping based on the source.
        """
        submodules = dict(pretrained_module.named_modules())
        if mapping is None:
            if "huggingface" in source:
                mapping = cls._huggingface_mapping
            else:
                mapping = {}
        inverse_mapping = {val: key for key, val in mapping.items()}
        for name, module in pretrained_module.named_modules():
            newname = name
            for key, val in inverse_mapping.items():
                newname = newname.replace(key, val)
            submodules[newname] = submodules.pop(name)
        return submodules

    def _construct_default_mapping(self, source):
        """
        Recursively constructs the default mapping of parameter names for loading pretrained module weights.
        Keys are parameter names from this module, and values are corresponding parameter names in the
        expected pretrained module, as per `source`.
        """
        mapping = {}
        if "huggingface" in source:
            mapping = self._huggingface_mapping
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

        pretrained_parameters = dict(pretrained_module.named_parameters())
        for name, parameter in self.named_parameters():
            pretrained_name = name
            for key, val in mapping.items():
                # so that we replace the names of submodules too.
                # eg. module.key.anothermodule --> module.val.anothermodule
                pretrained_name = pretrained_name.replace(key, val)
            if pretrained_name not in pretrained_parameters:
                raise ValueError(
                    f"Couldn't find a matching parameter for {name}. Is this module "
                    "compatible with the pretrained module you're using?"
                )
            parameter.data.copy_(pretrained_parameters[pretrained_name].data)

    @classmethod
    def _get_input_arguments(
        cls,
        pretrained_module: torch.nn.Module,
        source="huggingface",
        mapping: Optional[Dict[str, str]] = None,
    ):
        return NotImplementedError

    @classmethod
    def from_pretrained_module(
        cls,
        pretrained_module: torch.nn.Module,
        source="huggingface",
        mapping: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        final_kwargs = cls._get_input_arguments(pretrained_module, source, mapping)
        final_kwargs.update(kwargs)
        module = cls(**final_kwargs)
        module._load_from_pretrained_module(pretrained_module)
        return module
