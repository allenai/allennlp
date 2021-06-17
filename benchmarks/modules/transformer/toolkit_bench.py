import torch

from allennlp.modules.transformer import ActivationLayer


def bench_activation_layer(benchmark):
    layer = ActivationLayer(hidden_size=5, intermediate_size=3, activation="relu", pool=False)
    hidden_states = torch.randn(7, 5)
    benchmark(layer, hidden_states)


def bench_activation_layer_scripted(benchmark):
    layer = ActivationLayer(hidden_size=5, intermediate_size=3, activation="relu", pool=False)
    scripted = torch.jit.script(layer)
    hidden_states = torch.randn(7, 5)
    # the first time it runs, it adds a lot of overhead for some reason.
    scripted(hidden_states)
    benchmark(scripted, hidden_states)


def bench_activation_layer_traced(benchmark):
    layer = ActivationLayer(hidden_size=5, intermediate_size=3, activation="relu", pool=False)
    traced = torch.jit.trace(layer, torch.randn(7, 5))
    hidden_states = torch.randn(7, 5)
    benchmark(traced, hidden_states)
