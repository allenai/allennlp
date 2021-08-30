import torch

from allennlp.modules.transformer import SelfAttention

# from allennlp.modules.transformer.attention_module_script import SelfAttention as SelfAttentionScript


def bench_layer(benchmark):
    layer = SelfAttention(hidden_size=6, num_attention_heads=2, dropout=0.1)

    batch_size = 2
    seq_len = 3
    dim = layer.query.in_features
    hidden_states = torch.randn(batch_size, seq_len, dim)
    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 1]])[:, None, None, :]

    layer(hidden_states, attention_mask=attention_mask)
    benchmark(layer, hidden_states, attention_mask=attention_mask)


def bench_layer_scripted(benchmark):
    layer = SelfAttention(hidden_size=6, num_attention_heads=2, dropout=0.1)

    batch_size = 2
    seq_len = 3
    dim = layer.query.in_features
    hidden_states = torch.randn(batch_size, seq_len, dim)
    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 1]])[:, None, None, :]

    scripted = torch.jit.trace(layer, hidden_states)  # , attention_mask=attention_mask)
    # scripted(hidden_states, attention_mask=attention_mask)
    benchmark(scripted, hidden_states, attention_mask=attention_mask)
