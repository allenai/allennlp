import logging
import re
import warnings
from typing import Dict, NamedTuple, Optional, Tuple, Union, cast

import transformers
from allennlp.common.checks import ConfigurationError
from transformers import AutoConfig, AutoModel

logger = logging.getLogger(__name__)


class TransformerSpec(NamedTuple):
    model_name: str
    override_weights_file: Optional[str] = None
    override_weights_strip_prefix: Optional[str] = None
    reinit_modules: Optional[Union[int, Tuple[int, ...], Tuple[str, ...]]] = None


_model_cache: Dict[TransformerSpec, transformers.PreTrainedModel] = {}


def get(
    model_name: str,
    make_copy: bool,
    override_weights_file: Optional[str] = None,
    override_weights_strip_prefix: Optional[str] = None,
    reinit_modules: Optional[Union[int, Tuple[int, ...], Tuple[str, ...]]] = None,
    load_weights: bool = True,
    **kwargs,
) -> transformers.PreTrainedModel:
    """
    Returns a transformer model from the cache.

    # Parameters

    model_name : `str`
        The name of the transformer, for example `"bert-base-cased"`
    make_copy : `bool`
        If this is `True`, return a copy of the model instead of the cached model itself. If you want to modify the
        parameters of the model, set this to `True`. If you want only part of the model, set this to `False`, but
        make sure to `copy.deepcopy()` the bits you are keeping.
    override_weights_file : `str`, optional (default = `None`)
        If set, this specifies a file from which to load alternate weights that override the
        weights from huggingface. The file is expected to contain a PyTorch `state_dict`, created
        with `torch.save()`.
    override_weights_strip_prefix : `str`, optional (default = `None`)
        If set, strip the given prefix from the state dict when loading it.
    reinit_modules: `Optional[Union[int, Tuple[int, ...], Tuple[str, ...]]]`, optional (default = `None`)
        If this is an integer, the last `reinit_modules` layers of the transformer will be
        re-initialized. If this is a tuple of integers, the layers indexed by `reinit_modules` will
        be re-initialized. Note, because the module structure of the transformer `model_name` can
        differ, we cannot guarantee that providing an integer or tuple of integers will work. If
        this fails, you can instead provide a tuple of strings, which will be treated as regexes and
        any module with a name matching the regex will be re-initialized. Re-initializing the last
        few layers of a pretrained transformer can reduce the instability of fine-tuning on small
        datasets and may improve performance (https://arxiv.org/abs/2006.05987v3). Has no effect
        if `load_weights` is `False` or `override_weights_file` is not `None`.
    load_weights : `bool`, optional (default = `True`)
        If set to `False`, no weights will be loaded. This is helpful when you only
        want to initialize the architecture, like when you've already fine-tuned a model
        and are going to load the weights from a state dict elsewhere.
    """
    global _model_cache
    spec = TransformerSpec(
        model_name,
        override_weights_file,
        override_weights_strip_prefix,
        reinit_modules,
    )
    transformer = _model_cache.get(spec, None)
    if transformer is None:
        if not load_weights:
            if override_weights_file is not None:
                warnings.warn(
                    "You specified an 'override_weights_file' in allennlp.common.cached_transformers.get(), "
                    "but 'load_weights' is set to False, so 'override_weights_file' will be ignored.",
                    UserWarning,
                )
            if reinit_modules is not None:
                warnings.warn(
                    "You specified 'reinit_modules' in allennlp.common.cached_transformers.get(), "
                    "but 'load_weights' is set to False, so 'reinit_modules' will be ignored.",
                    UserWarning,
                )
            transformer = AutoModel.from_config(
                AutoConfig.from_pretrained(
                    model_name,
                    **kwargs,
                )
            )
        elif override_weights_file is not None:
            if reinit_modules is not None:
                warnings.warn(
                    "You specified 'reinit_modules' in allennlp.common.cached_transformers.get(), "
                    "but 'override_weights_file' is not None, so 'reinit_modules' will be ignored.",
                    UserWarning,
                )
            import torch
            from allennlp.common.file_utils import cached_path

            override_weights_file = cached_path(override_weights_file)
            override_weights = torch.load(override_weights_file)
            if override_weights_strip_prefix is not None:

                def strip_prefix(s):
                    if s.startswith(override_weights_strip_prefix):
                        return s[len(override_weights_strip_prefix) :]
                    else:
                        return s

                valid_keys = {
                    k
                    for k in override_weights.keys()
                    if k.startswith(override_weights_strip_prefix)
                }
                if len(valid_keys) > 0:
                    logger.info(
                        "Loading %d tensors from %s", len(valid_keys), override_weights_file
                    )
                else:
                    raise ValueError(
                        f"Specified prefix of '{override_weights_strip_prefix}' means no tensors "
                        f"will be loaded from {override_weights_file}."
                    )
                override_weights = {strip_prefix(k): override_weights[k] for k in valid_keys}

            transformer = AutoModel.from_config(
                AutoConfig.from_pretrained(
                    model_name,
                    **kwargs,
                )
            )
            # When DistributedDataParallel or DataParallel is used, the state dict of the
            # DistributedDataParallel/DataParallel wrapper prepends "module." to all parameters
            # of the actual model, since the actual model is stored within the module field.
            # This accounts for if a pretained model was saved without removing the
            # DistributedDataParallel/DataParallel wrapper.
            if hasattr(transformer, "module"):
                transformer.module.load_state_dict(override_weights)
            else:
                transformer.load_state_dict(override_weights)
        elif reinit_modules is not None:
            transformer = AutoModel.from_pretrained(
                model_name,
                **kwargs,
            )
            num_layers = transformer.config.num_hidden_layers
            if isinstance(reinit_modules, int):
                reinit_modules = tuple(range(num_layers - reinit_modules, num_layers))
            if all(isinstance(x, int) for x in reinit_modules):
                # This type cast is neccessary to avoid a mypy error.
                reinit_modules = cast(Tuple[int], reinit_modules)
                if any(layer_idx < 0 or layer_idx > num_layers for layer_idx in reinit_modules):
                    raise ValueError(
                        f"A layer index in reinit_modules ({reinit_modules}) is invalid."
                        f" Must be between 0 and the maximum layer index ({num_layers - 1}.)"
                    )
                # Some transformer models organize their modules differently, so if this fails,
                # raise an error with a helpful message.
                try:
                    for layer_idx in reinit_modules:
                        transformer.encoder.layer[layer_idx].apply(transformer._init_weights)
                except AttributeError:
                    raise ConfigurationError(
                        f"Unable to re-initialize the layers of transformer model"
                        f" {model_name} using layer indices. Please provide a tuple of"
                        " strings corresponding to the names of the layers to re-initialize."
                    )
            elif all(isinstance(x, str) for x in reinit_modules):
                for regex in reinit_modules:
                    for name, module in transformer.named_modules():
                        if re.search(str(regex), name):
                            module.apply(transformer._init_weights)
            else:
                raise ValueError(
                    "reinit_modules must be either an integer, a tuple of strings, or a tuple of integers."
                )
        else:
            transformer = AutoModel.from_pretrained(
                model_name,
                **kwargs,
            )
        _model_cache[spec] = transformer
    if make_copy:
        import copy

        return copy.deepcopy(transformer)
    else:
        return transformer


_tokenizer_cache: Dict[Tuple[str, str], transformers.PreTrainedTokenizer] = {}


def get_tokenizer(model_name: str, **kwargs) -> transformers.PreTrainedTokenizer:
    from allennlp.common.util import hash_object

    cache_key = (model_name, hash_object(kwargs))

    global _tokenizer_cache
    tokenizer = _tokenizer_cache.get(cache_key, None)
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            **kwargs,
        )
        _tokenizer_cache[cache_key] = tokenizer
    return tokenizer


def _clear_caches():
    """
    Clears in-memory transformer and tokenizer caches.
    """
    global _model_cache
    global _tokenizer_cache
    _model_cache.clear()
    _tokenizer_cache.clear()
