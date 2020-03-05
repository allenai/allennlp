from typing import Optional
import torch
import pytest

from allennlp.modules.seq2seq_encoders import PytorchTransformer


@pytest.mark.parametrize("positional_encoding", [None, "sinusoidal", "embedding"])
def test_positional_embeddings(positional_encoding: Optional[str]):
    # All sizes are prime, making them easy to find during debugging.
    batch_size = 7
    max_seq_len = 101
    n_head = 5
    dims = 11 * n_head
    transformer = PytorchTransformer(dims, 3, positional_encoding=positional_encoding, num_attention_heads=n_head)

    inputs = torch.randn(batch_size, max_seq_len, dims)
    mask = torch.ones(batch_size, max_seq_len, dtype=torch.bool)
    for b in range(batch_size):
        mask[b, max_seq_len-b:] = False

    outputs = transformer(inputs, mask)
    assert outputs.size() == inputs.size()
