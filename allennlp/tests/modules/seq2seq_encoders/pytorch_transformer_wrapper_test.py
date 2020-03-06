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
    transformer = PytorchTransformer(
        dims, 3, positional_encoding=positional_encoding, num_attention_heads=n_head
    )
    transformer.eval()

    with torch.no_grad():
        inputs = torch.randn(batch_size, max_seq_len, dims)
        mask = torch.ones(batch_size, max_seq_len, dtype=torch.bool)
        for b in range(batch_size):
            mask[b, max_seq_len - b :] = False

        assert not torch.isnan(inputs).any()
        assert torch.isfinite(inputs).all()
        outputs = transformer(inputs, mask)
        assert outputs.size() == inputs.size()
        assert not torch.isnan(outputs).any()
        assert torch.isfinite(outputs).all()


@pytest.mark.parametrize("positional_encoding", [None, "sinusoidal", "embedding"])
def test_mask_works(positional_encoding: Optional[str]):
    # All sizes are prime, making them easy to find during debugging.
    batch_size = 3
    max_seq_len = 11
    n_head = 2
    dims = 7 * n_head
    transformer = PytorchTransformer(
        dims, 2, positional_encoding=positional_encoding, num_attention_heads=n_head
    )
    transformer.eval()

    with torch.no_grad():
        # Construct inputs and masks
        inputs = torch.randn(batch_size, max_seq_len, dims)
        all_ones_mask = torch.ones(batch_size, max_seq_len, dtype=torch.bool)
        mask = all_ones_mask.clone()
        for b in range(batch_size):
            mask[b, max_seq_len - b :] = False
        altered_inputs = inputs + (~mask).unsqueeze(2) * 10.0

        # Make sure there is a difference without the mask
        assert not torch.allclose(
            transformer(inputs, all_ones_mask), transformer(altered_inputs, all_ones_mask)
        )

        # Make sure there is no difference with the mask
        assert torch.allclose(
            torch.masked_select(transformer(inputs, mask), mask.unsqueeze(2)),
            torch.masked_select(transformer(altered_inputs, mask), mask.unsqueeze(2)),
        )


@pytest.mark.parametrize("positional_encoding", [None, "sinusoidal", "embedding"])
def test_positional_encodings(positional_encoding: Optional[str]):
    # All sizes are prime, making them easy to find during debugging.
    batch_size = 3
    max_seq_len = 11
    n_head = 2
    dims = 7 * n_head
    transformer = PytorchTransformer(
        dims, 2, positional_encoding=positional_encoding, num_attention_heads=n_head
    )
    transformer.eval()

    with torch.no_grad():
        # We test this by running it twice, once with a shuffled sequence. The results should be the same if there
        # is no positional encoding, and different otherwise.
        inputs = torch.randn(batch_size, max_seq_len, dims)
        mask = torch.ones(batch_size, max_seq_len, dtype=torch.bool)
        for b in range(batch_size):
            mask[b, max_seq_len - b :] = False
        unshuffled_output = transformer(inputs, mask)

        shuffle = torch.arange(0, max_seq_len).unsqueeze(0).expand_as(mask).clone()
        for b in range(batch_size):
            # Take care not to shuffle the masked values
            perm = torch.randperm(max_seq_len - b)
            shuffle[b, : max_seq_len - b] = shuffle[b, perm]
        shuffle = shuffle.unsqueeze(2).expand_as(inputs)
        shuffled_input = torch.gather(inputs, 1, shuffle)
        shuffled_output = transformer(shuffled_input, mask)

        if positional_encoding is None:
            assert torch.allclose(
                torch.gather(unshuffled_output, 1, shuffle), shuffled_output, atol=2e-7
            )
        else:
            assert not torch.allclose(
                torch.gather(unshuffled_output, 1, shuffle), shuffled_output, atol=2e-7
            )
