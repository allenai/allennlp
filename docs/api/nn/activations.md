# allennlp.nn.activations

An :class:`Activation` is just a function
that takes some parameters and returns an element-wise activation function.
For the most part we just use
`PyTorch activations <https://pytorch.org/docs/master/nn.html#non-linear-activations>`_.
Here we provide a thin wrapper to allow registering them and instantiating them ``from_params``.

The available activation functions are

* "linear"
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

## Activation
```python
Activation(self, /, *args, **kwargs)
```

Pytorch has a number of built-in activation functions.  We group those here under a common
type, just to make it easier to configure and instantiate them ``from_params`` using
``Registrable``.

Note that we're only including element-wise activation functions in this list.  You really need
to think about masking when you do a softmax or other similar activation function, so it
requires a different API.

