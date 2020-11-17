import logging
from typing import NamedTuple, Optional, Dict, Tuple
import transformers
from transformers import AutoModel


logger = logging.getLogger(__name__)


class TransformerSpec(NamedTuple):
    model_name: str
    override_weights_file: Optional[str] = None
    override_weights_strip_prefix: Optional[str] = None


_model_cache: Dict[TransformerSpec, transformers.PreTrainedModel] = {}


def get(
    model_name: str,
    make_copy: bool,
    override_weights_file: Optional[str] = None,
    override_weights_strip_prefix: Optional[str] = None,
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
    override_weights_file : `str`, optional
        If set, this specifies a file from which to load alternate weights that override the
        weights from huggingface. The file is expected to contain a PyTorch `state_dict`, created
        with `torch.save()`.
    override_weights_strip_prefix : `str`, optional
        If set, strip the given prefix from the state dict when loading it.
    """
    global _model_cache
    spec = TransformerSpec(model_name, override_weights_file, override_weights_strip_prefix)
    transformer = _model_cache.get(spec, None)
    if transformer is None:
        if override_weights_file is not None:
            from allennlp.common.file_utils import cached_path
            import torch

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

            transformer = AutoModel.from_pretrained(
                model_name,
                state_dict=override_weights,
                **kwargs,
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
