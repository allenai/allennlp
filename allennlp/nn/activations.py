"""
An :class:`Activation` is just a function
that takes some parameters and returns an element-wise activation function.
For the most part we just use
`PyTorch activations <https://pytorch.org/docs/master/nn.html#non-linear-activations>`_.
Here we provide a thin wrapper to allow registering them and instantiating them `from_params`.

The available activation functions are

* "linear"
* `"mish" <https://arxiv.org/abs/1908.08681>`_
* `"swish" <https://arxiv.org/abs/1710.05941>`_
* `"relu" <https://pytorch.org/docs/master/nn.html#torch.nn.ReLU>`_
* `"relu6" <https://pytorch.org/docs/master/nn.html#torch.nn.ReLU6>`_
* `"elu" <https://pytorch.org/docs/master/nn.html#torch.nn.ELU>`_
* `"prelu" <https://pytorch.org/docs/master/nn.html#torch.nn.PReLU>`_
* `"leaky_relu" <https://pytorch.org/docs/master/nn.html#torch.nn.LeakyReLU>`_
* `"threshold" <https://pytorch.org/docs/master/nn.html#torch.nn.Threshold>`_
* `"hardtanh" <https://pytorch.org/docs/master/nn.html#torch.nn.Hardtanh>`_
* `"sigmoid" <https://pytorch.org/docs/master/nn.html#torch.nn.Sigmoid>`_
* `"tanh" <https://pytorch.org/docs/master/nn.html#torch.nn.Tanh>`_
* `"log_sigmoid" <https://pytorch.org/docs/master/nn.html#torch.nn.LogSigmoid>`_
* `"softplus" <https://pytorch.org/docs/master/nn.html#torch.nn.Softplus>`_
* `"softshrink" <https://pytorch.org/docs/master/nn.html#torch.nn.Softshrink>`_
* `"softsign" <https://pytorch.org/docs/master/nn.html#torch.nn.Softsign>`_
* `"tanhshrink" <https://pytorch.org/docs/master/nn.html#torch.nn.Tanhshrink>`_
* `"selu" <https://pytorch.org/docs/master/nn.html#torch.nn.SELU>`_
"""

import torch

from allennlp.common import Registrable


class Activation(Registrable):
    """
    Pytorch has a number of built-in activation functions.  We group those here under a common
    type, just to make it easier to configure and instantiate them `from_params` using
    `Registrable`.

    Note that we're only including element-wise activation functions in this list.  You really need
    to think about masking when you do a softmax or other similar activation function, so it
    requires a different API.
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        This function is here just to make mypy happy.  We expect activation functions to follow
        this API; the builtin pytorch activation functions follow this just fine, even though they
        don't subclass `Activation`.  We're just making it explicit here, so mypy knows that
        activations are callable like this.
        """
        raise NotImplementedError


# There are no classes to decorate, so we hack these into Registrable._registry.
# If you want to instantiate it, you can do like this:
# Activation.by_name('relu')()
Registrable._registry[Activation] = {
    "linear": (lambda: lambda x: x, None),  # type: ignore
    "mish": (lambda: lambda x: x * torch.tanh(torch.nn.functional.softplus(x)), None),  # type: ignore
    "swish": (lambda: lambda x: x * torch.sigmoid(x), None),  # type: ignore
    "relu": (torch.nn.ReLU, None),
    "relu6": (torch.nn.ReLU6, None),
    "elu": (torch.nn.ELU, None),
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
