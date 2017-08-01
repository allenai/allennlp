import torch

from allennlp.common import Registrable


class Activation(Registrable):
    """
    Pytorch has a number of built-in activation functions.  We group those here under a common
    type, just to make it easier to configure and instantiate them ``from_params`` using
    ``Registrable``.

    Note that we're only including element-wise activation functions in this list.  You really need
    to think about masking when you do a softmax or other similar activation function, so it
    requires a different API.
    """

# There are no classes to decorate, so we hack these into Registrable._registry
Registrable._registry[Activation] = {  # pylint: disable=protected-access
        "linear": lambda x: x,
        "relu": torch.nn.ReLU,
        "relu6": torch.nn.ReLU6,
        "elu": torch.nn.ELU,
        "selu": torch.nn.SELU,
        "prelu": torch.nn.PReLU,
        "leaky_relu": torch.nn.LeakyReLU,
        "threshold": torch.nn.Threshold,
        "hardtanh": torch.nn.Hardtanh,
        "sigmoid": torch.nn.Sigmoid,
        "tanh": torch.nn.Tanh,
        "log_sigmoid": torch.nn.LogSigmoid,
        "softplus": torch.nn.Softplus,
        "softshrink": torch.nn.Softshrink,
        "softsign": torch.nn.Softsign,
        "tanhshrink": torch.nn.Tanhshrink,
}
