"""
An implementation of the OpenAI Transformer Language Model.

Mostly just a slightly modified version of
https://github.com/huggingface/pytorch-openai-transformer-lm
so thanks to them!
"""

# pylint: disable=invalid-name,arguments-differ
from typing import NamedTuple, List
import copy
import io
import json
import logging
import math
import re
import tarfile

import numpy as np
import torch
from torch.nn import Parameter

from allennlp.common.file_utils import cached_path
from allennlp.modules.layer_norm import LayerNorm

logger = logging.getLogger(__name__)

def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

_ACTIVATION_FUNCTIONS = {
        'relu': torch.nn.ReLU,
        'swish': swish,
        'gelu': gelu
}

# pylint: disable=line-too-long
_PARAMETER_NAMES = [["model/we:0", "model/h0/attn/c_attn/w:0", "model/h0/attn/c_attn/b:0", "model/h0/attn/c_proj/w:0",
                     "model/h0/attn/c_proj/b:0", "model/h0/ln_1/g:0", "model/h0/ln_1/b:0", "model/h0/mlp/c_fc/w:0",
                     "model/h0/mlp/c_fc/b:0", "model/h0/mlp/c_proj/w:0", "model/h0/mlp/c_proj/b:0", "model/h0/ln_2/g:0",
                     "model/h0/ln_2/b:0", "model/h1/attn/c_attn/w:0", "model/h1/attn/c_attn/b:0", "model/h1/attn/c_proj/w:0",
                     "model/h1/attn/c_proj/b:0", "model/h1/ln_1/g:0", "model/h1/ln_1/b:0", "model/h1/mlp/c_fc/w:0",
                     "model/h1/mlp/c_fc/b:0", "model/h1/mlp/c_proj/w:0", "model/h1/mlp/c_proj/b:0", "model/h1/ln_2/g:0",
                     "model/h1/ln_2/b:0", "model/h2/attn/c_attn/w:0", "model/h2/attn/c_attn/b:0", "model/h2/attn/c_proj/w:0",
                     "model/h2/attn/c_proj/b:0", "model/h2/ln_1/g:0", "model/h2/ln_1/b:0", "model/h2/mlp/c_fc/w:0",
                     "model/h2/mlp/c_fc/b:0", "model/h2/mlp/c_proj/w:0", "model/h2/mlp/c_proj/b:0", "model/h2/ln_2/g:0",
                     "model/h2/ln_2/b:0", "model/h3/attn/c_attn/w:0", "model/h3/attn/c_attn/b:0", "model/h3/attn/c_proj/w:0",
                     "model/h3/attn/c_proj/b:0", "model/h3/ln_1/g:0", "model/h3/ln_1/b:0", "model/h3/mlp/c_fc/w:0",
                     "model/h3/mlp/c_fc/b:0", "model/h3/mlp/c_proj/w:0", "model/h3/mlp/c_proj/b:0", "model/h3/ln_2/g:0",
                     "model/h3/ln_2/b:0", "model/h4/attn/c_attn/w:0", "model/h4/attn/c_attn/b:0", "model/h4/attn/c_proj/w:0",
                     "model/h4/attn/c_proj/b:0", "model/h4/ln_1/g:0", "model/h4/ln_1/b:0", "model/h4/mlp/c_fc/w:0",
                     "model/h4/mlp/c_fc/b:0", "model/h4/mlp/c_proj/w:0", "model/h4/mlp/c_proj/b:0", "model/h4/ln_2/g:0",
                     "model/h4/ln_2/b:0", "model/h5/attn/c_attn/w:0", "model/h5/attn/c_attn/b:0", "model/h5/attn/c_proj/w:0",
                     "model/h5/attn/c_proj/b:0", "model/h5/ln_1/g:0", "model/h5/ln_1/b:0", "model/h5/mlp/c_fc/w:0",
                     "model/h5/mlp/c_fc/b:0", "model/h5/mlp/c_proj/w:0", "model/h5/mlp/c_proj/b:0", "model/h5/ln_2/g:0",
                     "model/h5/ln_2/b:0", "model/h6/attn/c_attn/w:0", "model/h6/attn/c_attn/b:0", "model/h6/attn/c_proj/w:0",
                     "model/h6/attn/c_proj/b:0", "model/h6/ln_1/g:0", "model/h6/ln_1/b:0", "model/h6/mlp/c_fc/w:0",
                     "model/h6/mlp/c_fc/b:0", "model/h6/mlp/c_proj/w:0", "model/h6/mlp/c_proj/b:0", "model/h6/ln_2/g:0",
                     "model/h6/ln_2/b:0", "model/h7/attn/c_attn/w:0", "model/h7/attn/c_attn/b:0", "model/h7/attn/c_proj/w:0",
                     "model/h7/attn/c_proj/b:0", "model/h7/ln_1/g:0", "model/h7/ln_1/b:0", "model/h7/mlp/c_fc/w:0",
                     "model/h7/mlp/c_fc/b:0", "model/h7/mlp/c_proj/w:0", "model/h7/mlp/c_proj/b:0", "model/h7/ln_2/g:0",
                     "model/h7/ln_2/b:0", "model/h8/attn/c_attn/w:0", "model/h8/attn/c_attn/b:0", "model/h8/attn/c_proj/w:0",
                     "model/h8/attn/c_proj/b:0", "model/h8/ln_1/g:0", "model/h8/ln_1/b:0", "model/h8/mlp/c_fc/w:0",
                     "model/h8/mlp/c_fc/b:0", "model/h8/mlp/c_proj/w:0", "model/h8/mlp/c_proj/b:0", "model/h8/ln_2/g:0",
                     "model/h8/ln_2/b:0", "model/h9/attn/c_attn/w:0", "model/h9/attn/c_attn/b:0", "model/h9/attn/c_proj/w:0",
                     "model/h9/attn/c_proj/b:0", "model/h9/ln_1/g:0", "model/h9/ln_1/b:0", "model/h9/mlp/c_fc/w:0",
                     "model/h9/mlp/c_fc/b:0", "model/h9/mlp/c_proj/w:0", "model/h9/mlp/c_proj/b:0", "model/h9/ln_2/g:0",
                     "model/h9/ln_2/b:0", "model/h10/attn/c_attn/w:0", "model/h10/attn/c_attn/b:0", "model/h10/attn/c_proj/w:0",
                     "model/h10/attn/c_proj/b:0", "model/h10/ln_1/g:0", "model/h10/ln_1/b:0", "model/h10/mlp/c_fc/w:0",
                     "model/h10/mlp/c_fc/b:0", "model/h10/mlp/c_proj/w:0", "model/h10/mlp/c_proj/b:0", "model/h10/ln_2/g:0",
                     "model/h10/ln_2/b:0", "model/h11/attn/c_attn/w:0", "model/h11/attn/c_attn/b:0", "model/h11/attn/c_proj/w:0",
                     "model/h11/attn/c_proj/b:0", "model/h11/ln_1/g:0", "model/h11/ln_1/b:0", "model/h11/mlp/c_fc/w:0",
                     "model/h11/mlp/c_fc/b:0", "model/h11/mlp/c_proj/w:0", "model/h11/mlp/c_proj/b:0", "model/h11/ln_2/g:0",
                     "model/h11/ln_2/b:0", "model/clf/w:0", "model/clf/b:0"]]
# pylint: enable=line-too-long


class TransformerConfig(NamedTuple):
    """
    If you want to load the weights of the pretrained
    OpenAI model, use the default parameters.
    """
    num_embeddings: int = 768
    num_heads: int = 12
    num_layers: int = 12
    embedding_dropout_probability: float = 0.1
    attention_dropout_probability: float = 0.1
    residual_dropout_probability: float = 0.1
    activation_function: str = 'gelu'
    classifier_dropout_probability: float = 0.1


class Conv1D(torch.nn.Module):
    def __init__(self,
                 nf: int,
                 rf: int,
                 nx: int) -> None:
        super().__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:
            w = torch.empty(nx, nf)
            torch.nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(nf))
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class Attention(torch.nn.Module):
    def __init__(self,
                 nx: int,
                 n_ctx: int,
                 config: TransformerConfig,
                 scale: bool = False):
        super().__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.num_heads == 0
        self.register_buffer('b', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.num_heads
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.attn_dropout = torch.nn.Dropout(config.attention_dropout_probability)
        self.resid_dropout = torch.nn.Dropout(config.residual_dropout_probability)

    def _attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        w = w * self.b + -1e9 * (1 - self.b)  # TF implem method: mask_attn_weights
        w = torch.nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x: torch.Tensor):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x: torch.Tensor, k: bool = False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a


class MLP(torch.nn.Module):
    def __init__(self, n_state: int, config: TransformerConfig):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        self.c_fc = Conv1D(n_state, 1, config.num_embeddings)
        self.c_proj = Conv1D(config.num_embeddings, 1, n_state)
        self.act = _ACTIVATION_FUNCTIONS[config.activation_function]
        self.dropout = torch.nn.Dropout(config.residual_dropout_probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(torch.nn.Module):
    def __init__(self,
                 n_ctx: int,
                 config: TransformerConfig,
                 scale: bool = False) -> None:
        super().__init__()
        nx = config.num_embeddings
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, config)
        self.ln_2 = LayerNorm(nx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.attn(x)
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        return h

class OpenaiTransformer(torch.nn.Module):
    """
    Openai transformer

    Parameters
    ----------

    """
    def __init__(self,
                 config: TransformerConfig = TransformerConfig(),
                 vocab_size: int = 40990,
                 n_ctx: int = 512,
                 requires_grad: bool = False) -> None:
        super().__init__()

        self.config = config
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.requires_grad = requires_grad

        self.embed = torch.nn.Embedding(vocab_size, config.num_embeddings)
        self.drop = torch.nn.Dropout(config.embedding_dropout_probability)

        block = Block(n_ctx, config, scale=True)
        self.h = torch.nn.ModuleList([copy.deepcopy(block) for _ in range(config.num_layers)])
        self.decoder = torch.nn.Linear(config.num_embeddings, vocab_size, bias=False)
        self.decoder.weight = self.embed.weight  # Tied weights
        # To reproduce the noise_shape parameter of TF implementation
        self.clf_dropout = torch.nn.Dropout2d(config.classifier_dropout_probability)

        torch.nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        #x = x.view(-1, x.size(2), x.size(3))

        with torch.set_grad_enabled(self.training and self.requires_grad):
            # x is (batch_size, sequence_length) tensor of byte-pair ids
            print("x", x.size())

            # e is (batch_size, sequence_length, 2, num_embeddings) tensor of embeddings
            e = self.embed(x)

            print("e", e.size())

            # h is (batch_size, sequence_length, num_embeddinggs)
            h = e.sum(dim=2)

            all_layers = [h]
            for block in self.h:
                h = block(h)
                all_layers.append(h)

            # result is list of (batch_size, sequence_length, num_embeddings)
            return all_layers

    def load_weights(self,
                     transformer_model_path: str,
                     n_ctx: int = -1,
                     n_special: int = -1,
                     n_transfer: int = 12,
                     n_embd: int = 768) -> None:

        logger.info(f"loading weights from {transformer_model_path}")
        # if `file_path` is a URL, redirect to the cache
        transformer_model_path = cached_path(transformer_model_path)

        names = _PARAMETER_NAMES

        with tarfile.open(transformer_model_path) as tmp:
            shapes = json.load(tmp.extractfile('model/params_shapes.json'))

            # numpy can't read from a tarfile directly, so we need a workaround
            # https://github.com/numpy/numpy/issues/7989#issuecomment-341656702
            init_params: List[np.ndarray] = []
            for n in range(10):
                array_file = io.BytesIO()
                array_file.write(tmp.extractfile(f'model/params_{n}.npy').read())
                array_file.seek(0)
                init_params.append(np.load(array_file))

        offsets = np.cumsum([np.prod(shape) for shape in shapes])
        init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
        init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
        if n_ctx > 0:
            init_params[0] = init_params[0][:n_ctx]
        if n_special > 0:
            init_params[0] = np.concatenate(
                    [init_params[1],
                     (np.random.randn(n_special, n_embd) * 0.02).astype(np.float32),
                     init_params[0]],
                    0
            )
        else:
            init_params[0] = np.concatenate([init_params[1], init_params[0]], 0)
        del init_params[1]
        if n_transfer == -1:
            n_transfer = 0
        else:
            n_transfer = 1 + n_transfer * 12
        init_params = [arr.squeeze() for arr in init_params]

        try:
            assert self.embed.weight.shape == init_params[0].shape
        except AssertionError as e:
            e.args += (self.embed.weight.shape, init_params[0].shape)
            raise

        self.embed.weight.data = torch.from_numpy(init_params[0])

        for name, ip in zip(names[1:n_transfer], init_params[1:n_transfer]):
            name = name[6:]  # skip "model/"
            assert name[-2:] == ":0"
            name = name[:-2]
            name = name.split('/')
            pointer = self
            for m_name in name:
                if re.fullmatch(r'[A-Za-z]+\d+', m_name):
                    l = re.split(r'(\d+)', m_name)
                else:
                    l = [m_name]
                pointer = getattr(pointer, l[0])
                if len(l) >= 2:
                    num = int(l[1])
                    pointer = pointer[num]
            try:
                assert pointer.shape == ip.shape
            except AssertionError as e:
                e.args += (pointer.shape, ip.shape)
                raise
            pointer.data = torch.from_numpy(ip)  # pylint: disable=attribute-defined-outside-init
