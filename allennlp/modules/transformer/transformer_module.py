from collections import OrderedDict
from enum import Enum
from itertools import chain
import logging
import os
from os import PathLike
from typing import TYPE_CHECKING, Optional, Dict, Union, List, Any, TypeVar, Type
import warnings

import torch
import torch.distributed as dist

from allennlp.common.util import is_distributed, is_global_primary
from allennlp.nn.util import distributed_device

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig


logger = logging.getLogger(__name__)


_T = TypeVar("_T", bound="TransformerModule")
StateDictType = Union[Dict[str, torch.Tensor], "OrderedDict[str, torch.Tensor]"]


class DistributedLoadingStrategy(Enum):
    """
    Strategy options for loading state dictionaries in distributed process groups.
    """

    FREE_FOR_ALL = "FREE_FOR_ALL"
    """
    Each process group loads its own state dict from disk.
    """

    MEMORY_EFFICIENT = "MEMORY_EFFICIENT"
    """
    Only the primary process group loads the state dict from disk, then it broadcasts
    each state tensor one-by-one to the other process groups.
    """

    @classmethod
    def from_str(cls, s: str) -> "DistributedLoadingStrategy":
        for option in cls:
            if option.value.lower() == s.lower():
                return option
        raise ValueError(f"Unknown distributed loading strategy: '{s}'")


class TransformerModule(torch.nn.Module):
    """
    Base class to help with generalized loading of pretrained weights.

    Subclasses should override `_from_config()` if you want to instantiate them with
    `from_pretrained_module()`.
    """

    _huggingface_mapping: Dict[str, str] = {}
    """
    An optional mapping for each class that determines any differences in the module
    names between the class modules and the HuggingFace model's modules.
    Keys correspond to HuggingFace submodule names, values correspond to submodules names of this module.
    """

    _relevant_module: Optional[Union[str, List[str]]] = None
    """
    An optional string or list of strings which contains the expected name of the module
    in the HuggingFace pretrained model. It can be a list to account for different names in different
    models. The search is carried out in the order of the list.
    """

    _distributed_loading_strategy: DistributedLoadingStrategy = (
        DistributedLoadingStrategy.FREE_FOR_ALL
    )
    """
    The default strategy for loading a state dictionary within a distributed process group.
    """

    @classmethod
    def _get_mapping(
        cls,
        mapping: Optional[Dict[str, str]] = None,
    ):
        """
        Returns the mapping to be used, based on the optional `mapping` overrides
        and the default module-level mapping.
        """
        combined_mapping = {}
        combined_mapping.update(cls._huggingface_mapping)
        if mapping is not None:
            combined_mapping.update(mapping)
        return combined_mapping

    @staticmethod
    def _get_mapped_state_dict(
        module: torch.nn.Module,
        state_dict: StateDictType,
        mapping: Optional[Dict[str, str]] = None,
    ) -> StateDictType:
        """
        Recursively map keys in a HuggingFace `state_dict` to the corresponding keys
        for this module and all submodules.

        This is a `@staticmethod` instead of an instance method so that we can call
        it on modules that do not inherit from `TransformerModule` in case those
        modules have submodules that are `TransformerModule` instances.
        """
        # First fix all top-level keys according to `combined_mapping`.
        combined_mapping = (
            module._get_mapping(mapping) if isinstance(module, TransformerModule) else {}
        )
        for hf_key, cls_key in combined_mapping.items():
            relevant_keys = set([key for key in state_dict.keys() if key.startswith(hf_key)])
            for key in relevant_keys:
                new_key = key.replace(hf_key, cls_key, 1)
                state_dict[new_key] = state_dict.pop(key)

        # Now loop through the submodules, calling this function on each submodule.
        for name, submodule in module.named_children():
            # Pull-out the part of the state_dict corresponding to just this submodule.
            relevant_keys = set([key for key in state_dict.keys() if key.startswith(name + ".")])
            module_state_dict = {
                key.replace(name + ".", "", 1): state_dict.pop(key) for key in relevant_keys
            }
            # Recursively call this function from the submodule to map this part
            # of the state_dict.
            module_state_dict = TransformerModule._get_mapped_state_dict(
                submodule, module_state_dict
            )
            # And then update the full state_dict.
            for key, value in module_state_dict.items():
                state_dict[name + "." + key] = value

        return state_dict

    @classmethod
    def _get_relevant_submodule_state(
        cls,
        state_dict: StateDictType,
        relevant_module: Optional[Union[str, List[str]]] = None,
    ) -> StateDictType:
        """
        Returns the relevant part of the `state_dict`.
        """
        relevant_modules: Optional[List[str]] = None
        if relevant_module:
            relevant_modules = (
                [relevant_module] if isinstance(relevant_module, str) else relevant_module
            )
        elif isinstance(cls._relevant_module, str):
            relevant_modules = [cls._relevant_module]
        elif isinstance(cls._relevant_module, list):
            relevant_modules = cls._relevant_module

        if relevant_modules:
            found = False
            for module_name in relevant_modules:
                relevant_keys = set(
                    [key for key in state_dict.keys() if key.startswith(module_name + ".")]
                )
                if relevant_keys:
                    # Only keep elements of state dict that correspond to the relevant module.
                    state_dict = {
                        key.replace(module_name + ".", "", 1): value
                        for key, value in state_dict.items()
                        if key in relevant_keys
                    }
                    found = True
                    break

            if not found:
                warnings.warn(
                    f"{relevant_modules} was not found at top level of state_dict!", UserWarning
                )

        return state_dict

    @classmethod
    def _get_pretrained_state_dict(
        cls,
        model_name: str,
        weights_path: Optional[Union[str, PathLike]] = None,
        relevant_module: Optional[Union[str, List[str]]] = None,
    ) -> StateDictType:
        """
        Get a HuggingFace pretrained `state_dict` corresponding to this module.
        """
        if weights_path is None:
            from transformers.file_utils import WEIGHTS_NAME

            # First see if we can find the weights locally.
            if os.path.isdir(model_name):
                local_weights_path = os.path.join(model_name, WEIGHTS_NAME)
                if os.path.isfile(local_weights_path):
                    logger.info("Found weights at local path %s", local_weights_path)
                    weights_path = local_weights_path

            # If we haven't found locally, we assume model ID corresponds to a model
            # on the HuggingFace Hub.
            if weights_path is None:
                from allennlp.common.file_utils import cached_path

                weights_path = cached_path(f"hf://{model_name}/{WEIGHTS_NAME}")

        # Now load the state dict.
        logger.info("Loading state dict from %s", weights_path)
        state_dict = torch.load(weights_path, map_location="cpu")

        # Keep just the relevant_module, remove everything else.
        state_dict = cls._get_relevant_submodule_state(state_dict)

        return state_dict

    @staticmethod
    def _collect_state_dict(
        module: torch.nn.Module, state_dict: Optional[StateDictType], recurse: bool = True
    ) -> StateDictType:
        """
        Collect a module's state dict across distributed processes.
        """
        # This is the device we'll use for the broadcast operation.
        device = distributed_device()

        # Gather current state dict and prepare to iterator over it.
        # We iterate over this state dict instead of `state_dict` so we can be sure
        # that the order is consistent across processes.
        # We'll also update this state dict as we go and return it at the end.
        if recurse:
            current_state_dict = module.state_dict()
        else:
            # Only collect state of direct members, including both parameters and buffers.
            current_state_dict = OrderedDict(
                chain(
                    # Paramaters
                    ((n, p.data) for (n, p) in module.named_parameters(recurse=False)),
                    # Buffers
                    module.named_buffers(recurse=False),
                )
            )
        keys = list(current_state_dict.keys())

        for key in keys:
            tensor = current_state_dict[key]
            if is_global_primary():
                assert state_dict is not None
                if key in state_dict:
                    tensor = state_dict[key]
                else:
                    logger.warning(
                        f"Missing key {key} from state_dict (available keys: {list(state_dict.keys())})"
                    )
            tensor = tensor.to(device)
            dist.broadcast(tensor, 0)
            current_state_dict[key] = tensor

        return current_state_dict

    @staticmethod
    def _load_state_dict_distributed(
        module: torch.nn.Module, state_dict: Optional[StateDictType], strict: bool = True
    ) -> None:
        """
        Load a `state_dict` within a distributed process group.

        The `state_dict` may be `None` if the current process group is not the global primary,
        in which case it will gather the parameters from the global primary one-by-one.

        This is a `@staticmethod` instead of an instance method so that we can call
        it on modules that do not inherit from `TransformerModule` in case those
        modules have submodules that are `TransformerModule` instances.
        """
        submodules = dict(module.named_children())

        # If we've found a sharded module or there aren't any more submodules of the current module,
        # we collect the state_dict and load it now instead of recursing further.
        if getattr(module, "_is_sharded", False) or not submodules:
            state_dict = TransformerModule._collect_state_dict(module, state_dict)
            assert state_dict is not None
            module.load_state_dict(state_dict, strict=strict)
        else:
            # We'll recursively call this function on each submodule, but first we need
            # to collect any parameters that are direct members of this module.
            direct_member_state_dict = TransformerModule._collect_state_dict(
                module, state_dict, recurse=False
            )
            missing_keys, unexpected_keys = module.load_state_dict(
                direct_member_state_dict, strict=False
            )
            if strict and unexpected_keys:
                raise ValueError(f"Unexpected keys in state dict: {unexpected_keys}")

            # Okay, now for the recursive part.
            for name, submodule in submodules.items():
                submodule_state_dict: Optional[StateDictType] = None
                if is_global_primary():
                    assert state_dict is not None
                    submodule_state_dict = {
                        key.replace(name + ".", "", 1): value
                        for key, value in state_dict.items()
                        if key.startswith(name + ".")
                    }
                submodule_state_dict = TransformerModule._collect_state_dict(
                    submodule, submodule_state_dict
                )
                assert submodule_state_dict is not None
                submodule.load_state_dict(submodule_state_dict, strict=strict)

    @classmethod
    def _from_config(cls: Type[_T], config: "PretrainedConfig", **kwargs) -> _T:
        """
        Instantiate this module from a HuggingFace config. Subclasses should override
        this method if you want to be able to instantiate them with `from_pretrained_module()`.
        """
        raise NotImplementedError

    @classmethod
    def from_pretrained_module(
        cls: Type[_T],
        model_name: str,
        load_weights: bool = True,
        weights_path: Optional[Union[str, PathLike]] = None,
        auto_config_kwargs: Optional[Dict[str, Any]] = None,
        mapping: Optional[Dict[str, str]] = None,
        relevant_module: Optional[Union[str, List[str]]] = None,
        strict: bool = True,
        distributed_loading_strategy: Optional[Union[str, DistributedLoadingStrategy]] = None,
        **kwargs,
    ) -> _T:
        """
        Initialize this module from a corresponding model from HuggingFace.


        !!! Note
            This method is only available for subclasses that implement `from_config()`.
            Otherwise a `NotImplementedError` will be raised.

        # Parameters

        model_name : `str`
            The model identifier or path.

        load_weights : `bool`, optional (default = `True`)
            Whether to download and load the pretrained weights. If `False`, the
            weights are left uninitialized.

        weights_path : `Optional[Union[str, PathLike]]`, optional (default = `None`)
            When `load_weights` is `True`, this can be set to override the weights file.
            Otherwise the default weights from the pretrained model are used.

        auto_config_kwargs : `Optional[Dict[str, Any]]`, optional (default = `None`)
            Optional key-word arguments to pass to `transformers.AutoConfig.from_pretrained()`
            to load the pretrained model's configuration file.

        mapping : `Optional[Dict[str, str]]`, optional (default = `None`)
            Optional mapping that determines any differences in the submodule names
            between this module and the pretrained model from HuggingFace.
            If not given, the class's default is used: `cls._huggingface_mapping`.

        relevant_module : `Optionall[str]`, optional (default = `None`)
            An optional submodule of the HuggingFace module to initialize weights from.
            This is only relevant when `load_weights` is `True`.
            If not given, the class's default is used: `cls._relevant_module`.

        strict : `bool`, optional (default = `True`)
            Whether to load the `state_dict` in "strict" model. This only applies
            when `load_weights` is `True`.

        distributed_loading_strategy : `Optional[Union[str, DistributedLoadingStrategy]]`, optional (default = `None`)
            The loading strategy to use within a distributed process group. This only applies
            when `load_weights` is `True`. If not specified, this class's default is used:
            `cls._distributed_loading_strategy`.

        **kwargs : Any
            Key word arguments to pass to `cls.from_config()` when instantiating the module.
        """  # noqa: E501
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, **(auto_config_kwargs or {}))
        model = cls._from_config(config, **kwargs)

        if load_weights:
            # Resolve the loading strategy to use.
            loading_strategy: DistributedLoadingStrategy
            if isinstance(distributed_loading_strategy, DistributedLoadingStrategy):
                loading_strategy = distributed_loading_strategy
            elif isinstance(distributed_loading_strategy, str):
                loading_strategy = DistributedLoadingStrategy.from_str(distributed_loading_strategy)
            else:
                loading_strategy = cls._distributed_loading_strategy

            state_dict: Optional[StateDictType] = None
            if is_global_primary() or loading_strategy == DistributedLoadingStrategy.FREE_FOR_ALL:
                # Load the pretrained HuggingFace state_dict.
                pretrained_state_dict = cls._get_pretrained_state_dict(
                    model_name,
                    weights_path=weights_path,
                    relevant_module=relevant_module,
                )
                # Now map keys from the HuggingFace state_dict to the corresponding keys from
                # this class. This is called recursively on each submodule of the current module.
                state_dict = TransformerModule._get_mapped_state_dict(
                    model, pretrained_state_dict, mapping=mapping
                )

            if not is_distributed() or loading_strategy == DistributedLoadingStrategy.FREE_FOR_ALL:
                assert state_dict is not None
                logger.info("Loading state_dict into module")
                model.load_state_dict(state_dict, strict=strict)
            else:
                # We're in distributed training. `state_dict` is `None` for all process groups
                # except the global primary.
                # Syncronize here since non-primary process groups will have to wait for the primary
                # to load the state_dict into memory.
                dist.barrier()
                # Now load the state dict into the model.
                logger.info("Loading state_dict into module (MEMORY_EFFICIENT strategy)")
                TransformerModule._load_state_dict_distributed(model, state_dict, strict=strict)

        return model
