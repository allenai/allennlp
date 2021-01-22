from typing import Optional, Dict, Union, List
import logging
import inspect

import torch

from allennlp.common import cached_transformers

logger = logging.getLogger(__name__)


class TransformerModule(torch.nn.Module):
    """
    Base class to help with generalized loading of pretrained weights.

    `_huggingface_mapping` is an optional mapping for each class, that determines
    any differences in the module names between the class modules and the huggingface model's
    modules.

    `_relevant_module` is an optional str or list of str which contains the expected name of the module
    in the huggingface pretrained model. It can be a list to account for different names in different
    models. The search is carried out in the order of the list.
    """

    _huggingface_mapping: Dict[str, str] = {}
    _relevant_module: Optional[Union[str, List[str]]] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _get_mapping(
        cls,
        pretrained_module: Optional[torch.nn.Module] = None,
        source="huggingface",
        mapping: Optional[Dict[str, str]] = None,
    ):
        """
        Returns the mapping to be used, based on the optional `pretrained_module`.
        If `pretrained_module` is not given, the default module-level mapping is returned.
        """
        combined_mapping = {}
        if "huggingface" in source:
            combined_mapping.update(cls._huggingface_mapping)
        if mapping is not None:
            combined_mapping.update(mapping)
        return combined_mapping

    @classmethod
    def _get_mapped_submodules(
        cls, pretrained_module, source="huggingface", mapping: Optional[Dict[str, str]] = None
    ):
        """
        Subclasses overload this method, and provide appropriate name mapping based on the source.
        """
        submodules = dict(pretrained_module.named_modules())
        combined_mapping = cls._get_mapping(pretrained_module, source, mapping)
        for name, module in pretrained_module.named_modules():
            newname = name
            for key, val in combined_mapping.items():
                newname = newname.replace(key, val)
            submodules[newname] = submodules.pop(name)
        return submodules

    def _construct_default_mapping(
        self,
        pretrained_module,
        source: str = "huggingface",
        mapping: Optional[Dict[str, str]] = None,
    ):
        """
        Recursively constructs the default mapping of parameter names for loading pretrained module weights.
        Keys are parameter names from this module, and values are corresponding parameter names in the
        expected pretrained module, as per `source`.
        """
        combined_mapping = self._get_mapping(pretrained_module, source, mapping)
        for name, module in self.named_modules():
            if name != "":
                if hasattr(module, "_construct_default_mapping"):
                    # We handle collisions by giving priority to the outer module's mapping.
                    combined_mapping = dict(
                        list(
                            module._construct_default_mapping(
                                pretrained_module, source, combined_mapping
                            ).items()
                        )
                        + list(combined_mapping.items())
                    )
        return combined_mapping

    def _load_from_pretrained_module(
        self,
        pretrained_module: torch.nn.Module,
        source="huggingface",
        mapping: Optional[Dict[str, str]] = None,
        ignore_absent_parameters: Optional[List] = None,
    ):
        """
        Loads the weights of the `pretrained_module` into the instance.
        Optionally, a `mapping` is specified for any differences in parameter names
        between `pretrained_module` and the instance.
        """
        ignore_absent_parameters = ignore_absent_parameters or []
        combined_mapping = self._construct_default_mapping(pretrained_module, source, mapping)
        if mapping is not None:
            combined_mapping.update(mapping)

        inverse_mapping = {val: key for key, val in combined_mapping.items()}
        pretrained_parameters = dict(pretrained_module.named_parameters())
        for name, parameter in self.named_parameters():
            pretrained_name = name
            for key, val in inverse_mapping.items():
                # so that we replace the names of submodules too.
                # eg. module.key.anothermodule --> module.val.anothermodule
                pretrained_name = pretrained_name.replace(key, val)
            if not any(
                [pretrained_name.startswith(paraname) for paraname in ignore_absent_parameters]
            ):
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
        **kwargs,
    ):
        """
        Constructs the arguments required for instantiating an object of this class, using
        the values from `pretrained_module`.
        """
        return kwargs

    @classmethod
    def get_relevant_module(
        cls,
        pretrained_module: Union[str, torch.nn.Module],
        relevant_module: Optional[Union[str, List[str]]] = None,
        source="huggingface",
        mapping: Optional[Dict[str, str]] = None,
    ):
        """
        Returns the relevant underlying module given a model name/object.

        # Parameters:

        pretrained_module: Name of the transformer model containing the layer,
                           or the actual layer (not the model object).
        relevant_module: Name of the desired module. Defaults to cls._relevant_module.
        source: Where the model came from. Default - huggingface.
        mapping: Optional mapping that determines any differences in the module names
        between the class modules and the input model's modules. Default - cls._huggingface_mapping
        """
        if isinstance(pretrained_module, str):
            pretrained_module = cached_transformers.get(pretrained_module, False)

        relevant_module = relevant_module or cls._relevant_module

        if relevant_module is not None:
            submodules = cls._get_mapped_submodules(pretrained_module, source, mapping)
            # If the relevant_module is not found, we assume that the pretrained_module
            # is already the relevant module.
            if isinstance(relevant_module, str):
                relevant_module = [relevant_module]
            found = False
            for module in relevant_module:
                if module in submodules:
                    pretrained_module = submodules[module]
                    found = True
                    break

            if not found:
                logger.warning(
                    "{} was not found! The submodules are: {}".format(
                        relevant_module, submodules.keys()
                    )
                )
        return pretrained_module

    @classmethod
    def from_pretrained_module(
        cls,
        pretrained_module: Union[str, torch.nn.Module],
        source="huggingface",
        mapping: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        Creates and returns an instance of the class, by using the weights
        (and the architecture, by default) of the `pretrained_module`.
        Optionally, the architecture can be changed by providing arguments.
        """
        accepted_args = inspect.getfullargspec(cls).args
        accepted_args.remove("self")
        for key in kwargs:
            assert key in accepted_args, (
                "{} is not a valid argument for creating an instance of `{}`. "
                "Accepted arguments are {}.".format(key, cls.__name__, accepted_args)
            )

        pretrained_module = cls.get_relevant_module(
            pretrained_module, source=source, mapping=mapping
        )
        final_kwargs = cls._get_input_arguments(pretrained_module, source, mapping)
        final_kwargs.update(kwargs)
        module = cls(**final_kwargs)
        module._load_from_pretrained_module(pretrained_module, source, mapping)
        return module
