import logging
import os
from os import PathLike
from typing import TYPE_CHECKING, Optional, Dict, Union, List, Any, TypeVar, Type
import re
import warnings

import torch
import torch.distributed as dist

from allennlp.common.util import is_distributed, is_global_primary
from allennlp.nn.util import StateDictType, read_state_dict, load_state_dict_distributed

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig


logger = logging.getLogger(__name__)


_T = TypeVar("_T", bound="TransformerModule")


class TransformerModule(torch.nn.Module):
    """
    Base class to help with generalized loading of pretrained weights.

    Subclasses should override `_from_config()` if you want to instantiate them with
    `from_pretrained_module()`.
    """

    _pretrained_mapping: Dict[str, str] = {}
    """
    An optional mapping for each class that determines any differences in the module
    names between the class modules and the HuggingFace model's modules.
    Keys correspond to HuggingFace submodule names, values correspond to submodules names of this module.
    """

    _pretrained_relevant_module: Optional[Union[str, List[str]]] = None
    """
    An optional string or list of strings which contains the expected name of the module
    in the HuggingFace pretrained model. It can be a list to account for different names in different
    models. The search is carried out in the order of the list.
    """

    _pretrained_ignore: Optional[List[str]] = None
    """
    An optional list of regular expressions that define which weights to ignore from a pretrained state_dict.
    """

    _pretrained_allow_missing: Optional[List[str]] = None
    """
    An optional list of regular expressions that specifies which weights are allowed to be missing
    from a pretrained state dictionary.
    """

    _tied_weights: Optional[Dict[str, List[str]]] = None
    """
    A mapping that defines any weights that need to be tied. Keys and values are parameter names.
    The values will be tied to the corresponding key.
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
        combined_mapping.update(cls._pretrained_mapping)
        if mapping is not None:
            combined_mapping.update(mapping)
        return combined_mapping

    def _get_mapped_state_dict(
        self,
        state_dict: StateDictType,
        mapping: Optional[Dict[str, str]] = None,
    ) -> StateDictType:
        """
        Recursively map keys in a HuggingFace `state_dict` to the corresponding keys
        for this module and all submodules.
        """
        return _get_mapped_state_dict(self, state_dict, mapping=mapping)

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
        elif isinstance(cls._pretrained_relevant_module, str):
            relevant_modules = [cls._pretrained_relevant_module]
        elif isinstance(cls._pretrained_relevant_module, list):
            relevant_modules = cls._pretrained_relevant_module

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
        ignore: Optional[List[str]] = None,
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
        logger.info("Reading state dict from %s", weights_path)
        state_dict = read_state_dict(
            weights_path,
            ignore=ignore if ignore is not None else cls._pretrained_ignore,
            strict=False,
        )

        # Keep just the relevant_module, remove everything else.
        state_dict = cls._get_relevant_submodule_state(state_dict, relevant_module=relevant_module)

        return state_dict

    @classmethod
    def _from_config(cls: Type[_T], config: "PretrainedConfig", **kwargs) -> _T:
        """
        Instantiate this module from a HuggingFace config. Subclasses should override
        this method if you want to be able to instantiate them with `from_pretrained_module()`.
        """
        raise NotImplementedError

    def tie_weights(self) -> None:
        """
        Tie weights according to the `_tied_weights` class attribute.

        This should always be called after loading a state dictionary. It will be called
        automatically within `from_pretrained_module()`.
        """
        if self._tied_weights:
            param_dict = dict(self.named_parameters())
            param_dict.update(dict(self.named_buffers()))
            for anchor_name, free_names in self._tied_weights.items():
                for free_name in free_names:
                    param_dict[free_name] = param_dict[anchor_name]

    @classmethod
    def from_pretrained_module(
        cls: Type[_T],
        model_name: str,
        *,
        load_weights: bool = True,
        weights_path: Optional[Union[str, PathLike]] = None,
        auto_config_kwargs: Optional[Dict[str, Any]] = None,
        mapping: Optional[Dict[str, str]] = None,
        relevant_module: Optional[Union[str, List[str]]] = None,
        ignore: Optional[List[str]] = None,
        allow_missing: Optional[List[str]] = None,
        strict: bool = True,
        **kwargs,
    ) -> _T:
        """
        Initialize this module from a corresponding model on HuggingFace.

        !!! Note
            This method is only available for subclasses that implement `_from_config()`.
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
            If not given, the class's default is used: `cls._pretrained_mapping`.

        relevant_module : `Optional[str]`, optional (default = `None`)
            An optional submodule of the HuggingFace module to initialize weights from.
            This is only relevant when `load_weights` is `True`.
            If not given, the class's default is used: `cls._pretrained_relevant_module`.

        ignore : `Optional[List[str]]`, optional (default = `None`)
            An optional list of regular expressions that define which weights to ignore
            from a pretrained state_dict.
            This is only relevant when `load_weights` is `True`.
            If not specified, the class's default is used: `cls._pretrained_ignore`.

        allow_missing: `Optional[List[str]]`, optional (default = `None`)
            An optional list of regular expressions that specifies which weights are allowed to be missing
            from the pretrained state dictionary.
            This is only relevant when `load_weights` is `True`.
            If not specified, the class's default is used: `cls._pretrained_allow_missing`.

        strict : `bool`, optional (default = `True`)
            Whether to load the `state_dict` in "strict" model. This only applies
            when `load_weights` is `True`.

        **kwargs : `Any`
            Key word arguments to pass to `cls.from_config()` when instantiating the module.
        """  # noqa: E501
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, **(auto_config_kwargs or {}))
        model = cls._from_config(config, **kwargs)

        if load_weights:
            state_dict: Optional[StateDictType] = None
            if is_global_primary():
                # Load the pretrained HuggingFace state_dict.
                pretrained_state_dict = cls._get_pretrained_state_dict(
                    model_name,
                    weights_path=weights_path,
                    relevant_module=relevant_module,
                    ignore=ignore,
                )
                # Now map keys from the HuggingFace state_dict to the corresponding keys from
                # this class. This is called recursively on each submodule of the current module.
                state_dict = model._get_mapped_state_dict(pretrained_state_dict, mapping=mapping)

            missing_keys: List[str]
            unexpected_keys: List[str]
            error_msgs: List[str] = []
            if not is_distributed():
                assert state_dict is not None
                logger.info("Loading state_dict into module")
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            else:
                # We're in distributed training. `state_dict` is `None` for all process groups
                # except the global primary.
                # Syncronize here since non-primary process groups will have to wait for the primary
                # to load the state_dict into memory.
                dist.barrier()
                # Now load the state dict into the model.
                logger.info("Loading state_dict into module (MEMORY_EFFICIENT strategy)")
                missing_keys, unexpected_keys = load_state_dict_distributed(
                    model, state_dict, strict=False
                )

            # Exclude any keys in `missing_keys` that match with the `allow_missing`
            # regular expressions.
            if allow_missing is None:
                allow_missing = cls._pretrained_allow_missing
            if allow_missing:
                missing_keys = [
                    k for k in missing_keys if not any(re.match(p, k) for p in allow_missing)
                ]

            # Allow missing keys in state_dict for params that are going to be tied.
            for param_names in (model._tied_weights or {}).values():
                for param_name in param_names:
                    if param_name in missing_keys:
                        missing_keys.remove(param_name)

            if missing_keys:
                error_msgs.append(
                    "Missing key(s) in state_dict: {}".format(
                        ", ".join(f'"{k}"' for k in missing_keys)
                    )
                )
            if unexpected_keys:
                error_msgs.append(
                    "Unexpected key(s) in state_dict: {}".format(
                        ", ".join(f'"{k}"' for k in unexpected_keys)
                    )
                )

            if error_msgs and strict:
                raise RuntimeError(
                    "Error(s) in loading state_dict for {}:\n\t{}".format(
                        cls.__name__, "\n\t".join(error_msgs)
                    )
                )

            # If there were error messages but we're not loading in 'strict' mode,
            # we just issue warnings from the logger.
            for msg in error_msgs:
                logger.warning(msg)

        model.tie_weights()

        return model


def _get_mapped_state_dict(
    module: torch.nn.Module,
    state_dict: StateDictType,
    mapping: Optional[Dict[str, str]] = None,
) -> StateDictType:
    # First fix all top-level keys according to `combined_mapping`.
    combined_mapping = module._get_mapping(mapping) if isinstance(module, TransformerModule) else {}
    for hf_key, cls_key in sorted(
        # Sort by most specific key first.
        combined_mapping.items(),
        key=lambda x: x[0].count("."),
        reverse=True,
    ):
        relevant_keys = set(
            [key for key in state_dict.keys() if (key == hf_key or key.startswith(hf_key + "."))]
        )
        for key in relevant_keys:
            new_key = key.replace(hf_key, cls_key, 1)
            # We have to be careful not to overwrite an entry that we might have updated
            # on a previous iteration of this loop due to having a more specific key.
            if new_key not in state_dict:
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
        module_state_dict = _get_mapped_state_dict(submodule, module_state_dict)
        # And then update the full state_dict.
        for key, value in module_state_dict.items():
            state_dict[name + "." + key] = value

    return state_dict
