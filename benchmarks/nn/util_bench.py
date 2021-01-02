import torch

from allennlp.nn import util
from allennlp.common.testing import requires_gpu


@requires_gpu
def bench_add_sentence_boundary_token_ids(benchmark):
    device = torch.device("cuda")
    # shape: (32, 50)
    tensor = torch.tensor([[3] * 50] * 32, device=device)
    # shape: (32, 50)
    mask = torch.tensor([[True] * 50, [True] * 30 + [False] * 20] * 16, device=device)
    begin_token = 1
    end_token = 2
    benchmark(util.add_sentence_boundary_token_ids, tensor, mask, begin_token, end_token)


@requires_gpu
def bench_remove_sentence_boundaries(benchmark):
    device = torch.device("cuda")
    # shape: (32, 50, 1)
    tensor = torch.tensor([[3] * 50] * 32, device=device).unsqueeze(-1)
    # shape: (32, 50)
    mask = torch.tensor([[True] * 50, [True] * 30 + [False] * 20] * 16, device=device)
    benchmark(util.remove_sentence_boundaries, tensor, mask)


@requires_gpu
def bench_create_tensor_then_send_to_device(benchmark):
    device = torch.device("cuda:0")

    def create_tensor():
        return torch.rand((32, 50)).to(device)

    benchmark(create_tensor)


@requires_gpu
def bench_create_tensor_directly_on_device(benchmark):
    device = torch.device("cuda:0")

    def create_tensor():
        return torch.rand((32, 50), device=device)

    benchmark(create_tensor)
