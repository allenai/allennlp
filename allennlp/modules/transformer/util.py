from typing import Union
import torch


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
    mask = (
        ~(mask == 1)
    ) * -10e5  # -10e5 to ensure that the model also works in half-precision mode.
    return values + mask
