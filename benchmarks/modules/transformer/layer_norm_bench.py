import torch

from allennlp.modules.transformer import LayerNorm


def bench_layer_norm(benchmark):
    layer = LayerNorm(normalized_shape=(2, 3), eps=1e-05, elementwise_affine=True)
    hidden_states = torch.randn(5, 2, 3)
    benchmark(layer, hidden_states)


def bench_layer_norm_scripted(benchmark):
    layer = LayerNorm(normalized_shape=(2, 3), eps=1e-05, elementwise_affine=True)
    scripted = torch.jit.script(layer)
    hidden_states = torch.randn(5, 2, 3)
    # the first time it runs, it adds a lot of overhead for some reason.
    scripted(hidden_states)
    benchmark(scripted, hidden_states)


def bench_layer_norm_traced(benchmark):
    layer = LayerNorm(normalized_shape=(2, 3), eps=1e-05, elementwise_affine=True)
    traced = torch.jit.trace(layer, torch.randn(5, 2, 3))
    hidden_states = torch.randn(5, 2, 3)
    benchmark(traced, hidden_states)
