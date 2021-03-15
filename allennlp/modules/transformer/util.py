from typing import Union, Tuple
import torch

from allennlp.nn.util import min_value_of_dtype


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
    if len(mask.shape) == 2:
        # We create a 4D attention mask from a 2D tensor mask.
        # The shape is `batch_size x 1 x 1 x target_seq_len` which is broadcast
        # to `batch_size x num_attention_heads x source_seq_len x target_seq_len`
        mask = mask.unsqueeze(1).unsqueeze(2)
    # `mask==1` to convert float tensors.
    mask = (~(mask == 1)) * min_value_of_dtype(values.dtype)
    return values + mask


def invert_attention_mask(encoder_attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Invert an attention mask (e.g., switches 0. and 1.).
    """
    # Adapted from https://github.com/huggingface/transformers/blob/
    # 4c32f9f26e6a84f0d9843fec8757e6ce640bb44e/src/transformers/modeling_utils.py#L187.
    if encoder_attention_mask.dim() == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    if encoder_attention_mask.dim() == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

    encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype)

    # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
    # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
    # /transformer/transformer_layers.py#L270
    # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
    # encoder_extended_attention_mask.transpose(-1, -2))

    encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * min_value_of_dtype(
        dtype
    )

    return encoder_extended_attention_mask


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

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and a very negative number for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * min_value_of_dtype(dtype)
    return extended_attention_mask
