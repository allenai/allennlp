"""
An `Activation` is just a function
that takes some parameters and returns an element-wise activation function.
For the most part we just use
[PyTorch activations](https://pytorch.org/docs/master/nn.html#non-linear-activations).
Here we provide a thin wrapper to allow registering them and instantiating them `from_params`.

The available activation functions are

* "linear"
* ["mish"](https://arxiv.org/abs/1908.08681)
* ["swish"](https://arxiv.org/abs/1710.05941)
* ["relu"](https://pytorch.org/docs/master/nn.html#torch.nn.ReLU)
* ["relu6"](https://pytorch.org/docs/master/nn.html#torch.nn.ReLU6)
* ["elu"](https://pytorch.org/docs/master/nn.html#torch.nn.ELU)
* ["prelu"](https://pytorch.org/docs/master/nn.html#torch.nn.PReLU)
* ["leaky_relu"](https://pytorch.org/docs/master/nn.html#torch.nn.LeakyReLU)
* ["threshold"](https://pytorch.org/docs/master/nn.html#torch.nn.Threshold)
* ["hardtanh"](https://pytorch.org/docs/master/nn.html#torch.nn.Hardtanh)
* ["sigmoid"](https://pytorch.org/docs/master/nn.html#torch.nn.Sigmoid)
* ["tanh"](https://pytorch.org/docs/master/nn.html#torch.nn.Tanh)
* ["log_sigmoid"](https://pytorch.org/docs/master/nn.html#torch.nn.LogSigmoid)
* ["softplus"](https://pytorch.org/docs/master/nn.html#torch.nn.Softplus)
* ["softshrink"](https://pytorch.org/docs/master/nn.html#torch.nn.Softshrink)
* ["softsign"](https://pytorch.org/docs/master/nn.html#torch.nn.Softsign)
* ["tanhshrink"](https://pytorch.org/docs/master/nn.html#torch.nn.Tanhshrink)
* ["selu"](https://pytorch.org/docs/master/nn.html#torch.nn.SELU)
"""

import torch

from allennlp.common import Registrable


class Activation(torch.nn.Module, Registrable):
    """
    Pytorch has a number of built-in activation functions.  We group those here under a common
    type, just to make it easier to configure and instantiate them `from_params` using
    `Registrable`.

    Note that we're only including element-wise activation functions in this list.  You really need
    to think about masking when you do a softmax or other similar activation function, so it
    requires a different API.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# There are no classes to decorate, so we hack these into Registrable._registry.
# If you want to instantiate it, you can do like this:
# Activation.by_name('relu')()
Registrable._registry[Activation] = {
    "relu": (torch.nn.ReLU, None),
    "relu6": (torch.nn.ReLU6, None),
    "elu": (torch.nn.ELU, None),
    "gelu": (torch.nn.GELU, None),
    "prelu": (torch.nn.PReLU, None),
    "leaky_relu": (torch.nn.LeakyReLU, None),
    "threshold": (torch.nn.Threshold, None),
    "hardtanh": (torch.nn.Hardtanh, None),
    "sigmoid": (torch.nn.Sigmoid, None),
    "tanh": (torch.nn.Tanh, None),
    "log_sigmoid": (torch.nn.LogSigmoid, None),
    "softplus": (torch.nn.Softplus, None),
    "softshrink": (torch.nn.Softshrink, None),
    "softsign": (torch.nn.Softsign, None),
    "tanhshrink": (torch.nn.Tanhshrink, None),
    "selu": (torch.nn.SELU, None),
}


@Activation.register("linear")
class LinearActivation(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


@Activation.register("mish")
class MishActivation(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(torch.nn.functional.softplus(x))


@Activation.register("swish")
class SwishActivation(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
