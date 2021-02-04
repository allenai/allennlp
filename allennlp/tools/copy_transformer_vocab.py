from typing import Dict, Any
from transformers import PreTrainedTokenizer


def copy_transformer_vocab(tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
    """
    Copies tokens from ```transformers``` model's vocab
    """

    try:
        vocab_items = tokenizer.get_vocab().items()
    except NotImplementedError:
        vocab_items = (
            (tokenizer.convert_ids_to_tokens(idx), idx) for idx in range(tokenizer.vocab_size)
        )
    outputs = dict()
    outputs["token_to_index"] = dict()
    outputs["index_to_token"] = dict()
    for word, idx in vocab_items:
        outputs["token_to_index"][word] = idx
        outputs["index_to_token"][idx] = word

    return outputs
