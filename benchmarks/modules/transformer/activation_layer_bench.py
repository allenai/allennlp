import torch

from allennlp.modules.transformer import ActivationLayer

"""
hidden_size: int,
        intermediate_size: int,
        activation: Union[str, torch.nn.Module],
        pool: bool = False,
"""


def bench_output_layer(benchmark):
    layer = ActivationLayer(hidden_size=3, intermediate_size=5, activation="relu", pool=False)
    hidden_states = torch.randn(2, 3)
    benchmark(layer, hidden_states)


def bench_output_layer_scripted(benchmark):
    layer = ActivationLayer(hidden_size=3, intermediate_size=5, activation="relu", pool=False)
    hidden_states = torch.randn(2, 3)
    scripted = torch.jit.script(layer)
    # the first time it runs, it adds a lot of overhead for some reason.
    scripted(hidden_states)
    benchmark(scripted, hidden_states)


def bench_output_layer_traced(benchmark):
    layer = ActivationLayer(hidden_size=3, intermediate_size=5, activation="relu", pool=False)
    hidden_states = torch.randn(2, 3)
    traced = torch.jit.trace(layer, torch.randn(2, 3))
    benchmark(traced, hidden_states)
