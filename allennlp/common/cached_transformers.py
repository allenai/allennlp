from typing import NamedTuple, Optional, Dict
import transformers
from transformers import AutoModel


class TransformerSpec(NamedTuple):
    model_name: str
    override_weights_file: Optional[str] = None
    override_weights_strip_prefix: Optional[str] = None


_transformer_model_cache: Dict[TransformerSpec, transformers.PreTrainedModel] = {}


def get(
    model_name: str,
    override_weights_file: Optional[str] = None,
    override_weights_strip_prefix: Optional[str] = None,
):
    global _transformer_model_cache
    spec = TransformerSpec(model_name, override_weights_file, override_weights_strip_prefix)
    transformer = _transformer_model_cache.get(spec, None)
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

                override_weights = {strip_prefix(k): v for k, v in override_weights.items()}
            transformer = AutoModel.from_pretrained(model_name, state_dict=override_weights)
        else:
            transformer = AutoModel.from_pretrained(model_name)
        _transformer_model_cache[spec] = transformer
    return transformer
