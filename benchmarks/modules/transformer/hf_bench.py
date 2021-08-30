import torch
import copy
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertModel

import numpy
from time import perf_counter


def timer(f, *args):

    start = perf_counter()
    f(*args)
    return 1000 * (perf_counter() - start)


params = {
    "num_hidden_layers": 3,
    "hidden_size": 6,
    "intermediate_size": 3,
    "num_attention_heads": 2,
    "attention_probs_dropout_prob": 0.1,
    "hidden_dropout_prob": 0.2,
    "activation": "relu",
}

# def noscript_bench(benchmark):
#   params_dict = copy.deepcopy(params)
#   hf_module = BertEncoder(BertConfig(**params_dict))
#   hidden_states = torch.randn(2, 5, 6)
#   benchmark(hf_module, hidden_states)

# def torchscript_bench(benchmark):
#   params_dict = copy.deepcopy(params)
#   params_dict["torchscript"] = True
#   hf_module = BertEncoder(BertConfig(**params_dict))
#   hidden_states = torch.randn(2, 5, 6)

#   traced = torch.jit.script(hf_module) #, hidden_states, strict=False)
#   benchmark(traced, hidden_states)


def noscript_bench(benchmark):
    native_model = BertModel.from_pretrained("bert-base-uncased")  # .cuda()
    hidden_states = torch.LongTensor([[3, 4, 6, 5, 1], [6, 3, 1, 2, 5]])  # .cuda()
    benchmark(native_model, hidden_states)
    # print("noscript_bench", np.mean([timer(native_model, hidden_states) for _ in range(15)]))


def torchscript_bench(benchmark):
    script_model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)  # .cuda()
    hidden_states = torch.LongTensor([[3, 4, 6, 5, 1], [6, 3, 1, 2, 5]])  # .cuda()
    script_model = torch.jit.trace(script_model, hidden_states)
    # print("torchscript_bench", np.mean([timer(script_model, hidden_states) for _ in range(15)]))
    benchmark(script_model, hidden_states)
