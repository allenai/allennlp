from typing import Union, Tuple
import torch
from allennlp.nn.util import min_value_of_dtype

# Unfortunately mypy is insane, so we have to wrap these in unions.
FloatT = Union[torch.FloatTensor]
IntT = Union[torch.IntTensor]
BoolT = Union[torch.BoolTensor]


def apply_mask(
    values: torch.FloatTensor, mask: Union[torch.BoolTensor, torch.IntTensor, torch.FloatTensor]
) -> torch.FloatTensor:
    """
    # Parameters

    values : `torch.FloatTensor`
        Shape `batch_size x num_attention_heads x source_seq_len x target_seq_len`
    mask : `torch.BoolTensor`
        Shape `batch_size x target_seq_len` OR `batch_size x 1 x 1 x target_seq_len`
    """
    # We create a 4D attention mask from a 2D or 3D tensor mask.
    if mask.dim() == 2:
        # The shape is `batch_size x 1 x 1 x target_seq_len` which is broadcast
        # to `batch_size x num_attention_heads x source_seq_len x target_seq_len`
        mask = mask[:, None, None, :]
    elif mask.dim() == 3:
        mask = mask[:, None, :, :]
    mask = mask.to(values.dtype)
    mask = (1.0 - mask) * min_value_of_dtype(values.dtype)
    return values + mask


def get_extended_attention_mask(
    attention_mask: torch.Tensor,
    input_shape: Tuple[int, ...],
    dtype: torch.dtype,
    is_decoder: bool = False,
) -> torch.Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    # Parameters

    attention_mask : `torch.Tensor`
        Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
    input_shape : `Tuple[int, ...]`
        The shape of the input to the model.
    dtype : `torch.dtype`
        The datatype of the resulting mask.
    is_decoder : `bool`, optional (default = `False`)
        If this is for a decoder stack.

    # Returns

    `torch.Tensor`
        The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """
    # Adapted from https://github.com/huggingface/transformers/blob/
    # 4c32f9f26e6a84f0d9843fec8757e6ce640bb44e/src/transformers/modeling_utils.py#L221.

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to
        #   `(batch_size, num_heads, seq_length, seq_length)`
        if is_decoder:
            batch_size, seq_length = input_shape
            seq_ids = torch.arange(seq_length, device=attention_mask.device)
            causal_mask = (
                seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
            )
            # in case past_key_values are used we need to add a prefix ones mask to the causal mask
            # causal and attention masks must have same type with pytorch version < 1.3
            causal_mask = causal_mask.to(attention_mask.dtype)

            if causal_mask.shape[1] < attention_mask.shape[1]:
                prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                causal_mask = torch.cat(
                    [
                        torch.ones(
                            (batch_size, seq_length, prefix_seq_len),
                            device=attention_mask.device,
                            dtype=causal_mask.dtype,
                        ),
                        causal_mask,
                    ],
                    axis=-1,
                )

            extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                input_shape, attention_mask.shape
            )
        )

    return extended_attention_mask
