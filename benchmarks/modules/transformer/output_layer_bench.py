import torch

from allennlp.modules.transformer import OutputLayer


def bench_output_layer(benchmark):
    layer = OutputLayer(input_size=3, hidden_size=5, dropout=0.1)
    hidden_states = torch.randn(2, 3)
    input_tensor = torch.randn(2, 5)
    benchmark(layer, hidden_states, input_tensor)


def bench_output_layer_scripted(benchmark):
    layer = OutputLayer(input_size=3, hidden_size=5, dropout=0.1)
    hidden_states = torch.randn(2, 3)
    input_tensor = torch.randn(2, 5)
    scripted = torch.jit.script(layer)
    # the first time it runs, it adds a lot of overhead for some reason.
    scripted(hidden_states, input_tensor)
    benchmark(scripted, hidden_states, input_tensor)


def bench_output_layer_traced(benchmark):
    layer = OutputLayer(input_size=3, hidden_size=5, dropout=0.1)
    hidden_states = torch.randn(2, 3)
    input_tensor = torch.randn(2, 5)
    traced = torch.jit.trace(layer, (torch.randn(2, 3), torch.randn(2, 5)))
    benchmark(traced, hidden_states, input_tensor)
